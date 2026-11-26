// ng2 - Ninja Gaiden II static recompilation
//
// Native 120 FPS state machine and render-thread sub-frame isolation.

#include "high_fps/high_fps.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <thread>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/types.h>

#include "gpu/guest.h"
#include "gpu/video.h"
#include "gpu/win_display.h"
#include "high_fps/camera_120hz.h"
#include "high_fps/input_120hz.h"
#include "high_fps/state_bridge.h"
#include "movie/movie.h"

REXCVAR_DEFINE_BOOL(native_120fps, false, "Video",
                    "Native 120 FPS decoupled rendering (60 Hz logic, 120 Hz render thread)");
REXCVAR_DEFINE_INT32(frame_rate_mode, 0, "Video",
                     "0: 60 FPS (original), 1: 120 FPS (Fixed), 2: Variable (VFR)")
    .range(0, 2);
REXCVAR_DEFINE_INT32(vfr_fps_limit, 0, "Video",
                     "VFR frame rate cap: 0 unlimited/VSync, >0 target FPS, -1 Display - 3 (VRR)");

namespace ng2::high_fps {
namespace {

constexpr uint32_t kQuitFlag = 0x83C07844;
constexpr uint32_t kPostPasses = 0x84C55758;
constexpr uint32_t kFadeBase = 0x84C5580C;  // 8 words: 0x84C5580C..0x84C55828
constexpr uint32_t kFadeResetFlag = 0x84C5582C;
constexpr uint32_t kSceneActive = 0x859EEB64 + 1324;  // 0x859EF090
constexpr uint32_t kRenderMgrSlotParity = 0x859EEB64 + 1316;
constexpr uint32_t kStagePoseParity = 0x851E2320;
constexpr uint32_t kRenderCounter = 0x84C5572C;
constexpr uint32_t kRenderEndFlagA = 0x84C55B48;
constexpr uint32_t kRenderEndFlagB = 0x84C55B4C;
constexpr uint32_t kSwapCounter = 0x84C55B54;
constexpr uint32_t kVblankInterval = 0x84C5342C;
constexpr uint32_t kCutscene30fpsFlag = 0x84C555D8;
constexpr uint32_t kGammaDirty = 0x83C07820;
constexpr uint32_t kTileResetFlag = 0x859F0518;
constexpr uint32_t kRenderMgr = 0x859EEB64;

struct SubFrameSnapshot {
  uint32_t postPasses = 0;
  uint32_t fadeWords[8] = {};
  uint32_t fadeResetFlag = 0;
  uint32_t sceneActive = 0;
  uint32_t renderMgrSlotParity = 0;
  uint32_t stagePoseParity = 0;
  uint32_t renderCounter = 0;
  uint32_t renderEndFlagA = 0;
  uint32_t renderEndFlagB = 0;
  uint32_t swapCounter = 0;
  uint32_t vblankInterval = 0;
  uint32_t gammaDirty = 0;
  uint32_t tileResetFlag = 0;
  uint64_t ring0Count = 0;
  uint64_t ring1Count = 0;
  uint32_t ring0Addr = 0;
  uint32_t ring1Addr = 0;
};

void Store32(uint32_t address, uint32_t value) {
  *reinterpret_cast<uint32_t*>(ng2::guest::Virtual(address)) = rex::byte_swap(value);
}

uint64_t Load64(uint32_t address) {
  return rex::byte_swap(*reinterpret_cast<const uint64_t*>(ng2::guest::Virtual(address)));
}

void Store64(uint32_t address, uint64_t value) {
  *reinterpret_cast<uint64_t*>(ng2::guest::Virtual(address)) = rex::byte_swap(value);
}

std::atomic<bool> g_vblank120Active{false};
std::atomic<bool> g_overloadDegraded{false};
std::atomic<uint64_t> g_logicFrameId{0};
std::atomic<int> g_phase{int(Phase::kDisabled)};
std::atomic<int> g_frameMode{int(RateMode::kOff60)};
std::atomic<uint32_t> g_subFrameIndex{0};
std::atomic<uint32_t> g_subFrameCount{1};
std::atomic<float> g_subFrameAlpha{1.0f};
std::atomic<double> g_subFrameDeltaSec{1.0 / 60.0};
std::atomic<bool> g_is30FpsMode{false};
bool g_lastLogged30Fps = false;
RateMode g_lastLoggedRateMode = RateMode::kOff60;
bool g_rateModeLogged = false;

int64_t SteadyNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void VfrSleepUntilNs(int64_t targetNs) {
  int64_t waitNs = targetNs - SteadyNs();
  if (waitNs <= 0) {
    return;
  }
  constexpr int64_t kSpinNs = 300'000;
  if (waitNs > kSpinNs && ng2::video::PreciseSleepAvailable()) {
    ng2::video::PreciseSleepNs(waitNs - kSpinNs);
  } else if (waitNs > 1'500'000) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(waitNs - 1'500'000));
  }
  while (SteadyNs() < targetNs) {
    std::this_thread::yield();
  }
}

// Dynamic 120 Hz -> 60 Hz overload guardrail (render-thread only).
// Sub-frame wall duration includes VSync/FlipWait (~8.333 ms at 120 Hz), so
// thresholds must sit well above 8.333 ms to avoid false-triggering on healthy
// 120 Hz VSync-locked frames or 30 fps (R=4) frames where sub_822F0BF8 sleeps.
// When degraded, VsyncThread drops physical VBlank pacing back to 60 Hz in
// lockstep so R=1 NEVER runs against 120 Hz VBlanks (which would run at 2x
// game speed).
constexpr int64_t kOverloadAvgSubFrameNs = 10'500'000;   // 10.5 ms avg/sub-frame (< ~95 FPS)
constexpr int64_t kOverloadPeakSubFrameNs = 18'000'000;  // 18.0 ms severe stall peak
constexpr uint32_t kOverloadLogicFrames = 12;            // 12 consecutive 60 Hz logic frames
constexpr int64_t kRecovery60HzFrameNs = 17'500'000;     // healthy 60 Hz VSync wall time (<= 17.5 ms)
constexpr uint32_t kMinDegradedLogicFrames = 120;        // 2.0 s cooldown at 60 Hz
constexpr uint32_t kRecoveryLogicFrames = 30;            // 0.5 s steady recovery

int64_t g_logicRenderTotalNs = 0;
int64_t g_logicMaxSubFrameNs = 0;
uint32_t g_overloadStreak = 0;
uint32_t g_degradedLogicFrames = 0;
uint32_t g_recoveryStreak = 0;

bool OverloadGuardEnabled() {
  static const bool on = [] {
    const char* v = std::getenv("NG2_NATIVE_120FPS_NO_GUARD");
    return !(v && v[0] == '1');
  }();
  return on;
}

void ResetOverloadGuard() {
  g_overloadDegraded.store(false, std::memory_order_release);
  g_logicRenderTotalNs = 0;
  g_logicMaxSubFrameNs = 0;
  g_overloadStreak = 0;
  g_degradedLogicFrames = 0;
  g_recoveryStreak = 0;
}

bool ShouldUseR4For30Fps() {
  if (ng2::movie::AnyFullscreenPlaying()) {
    return false;
  }
  return ng2::guest::Load32(kVblankInterval) == 2 ||
         ng2::guest::Load32(kCutscene30fpsFlag) != 0;
}

void EvaluateLogicFrameLoad(int64_t totalNs, int64_t maxSubNs, uint32_t subFrames) {
  if (!OverloadGuardEnabled() || !Enabled() || subFrames == 0) {
    return;
  }
  if (!g_overloadDegraded.load(std::memory_order_acquire)) {
    // Only evaluate overload in normal 60 Hz -> 120 Hz gameplay (R=2); skip
    // 30 fps UI/cutscene (R=4) where sub_822F0BF8 deliberately sleeps to 33.3 ms.
    if (!IsActive() || subFrames != 2 || ShouldUseR4For30Fps()) {
      g_overloadStreak = 0;
      return;
    }
    const int64_t avgSubNs = totalNs / int64_t(subFrames);
    if (avgSubNs > kOverloadAvgSubFrameNs || maxSubNs > kOverloadPeakSubFrameNs) {
      if (++g_overloadStreak >= kOverloadLogicFrames) {
        g_overloadDegraded.store(true, std::memory_order_release);
        g_overloadStreak = 0;
        g_degradedLogicFrames = 0;
        g_recoveryStreak = 0;
        REXLOG_WARN(
            "high_fps: render overload ({:.2f} ms avg / {:.2f} ms peak over {} frames), "
            "degrading to 60 Hz (R=1)",
            double(avgSubNs) / 1e6, double(maxSubNs) / 1e6, kOverloadLogicFrames);
      }
    } else {
      g_overloadStreak = 0;
    }
  } else {
    // While degraded, VsyncThread runs at 60 Hz (~16.67 ms per logic frame).
    if (Vblank120Active()) {
      return;  // Wait for VsyncThread to finish switching down to 60 Hz.
    }
    g_degradedLogicFrames++;
    if (maxSubNs <= kRecovery60HzFrameNs) {
      g_recoveryStreak++;
      if (g_degradedLogicFrames >= kMinDegradedLogicFrames &&
          g_recoveryStreak >= kRecoveryLogicFrames) {
        g_overloadDegraded.store(false, std::memory_order_release);
        g_overloadStreak = 0;
        g_degradedLogicFrames = 0;
        g_recoveryStreak = 0;
        REXLOG_INFO("high_fps: render load recovered ({:.2f} ms at 60 Hz), restoring 120 Hz (R=2)",
                    double(maxSubNs) / 1e6);
      }
    } else {
      g_recoveryStreak = 0;
    }
  }
}

// Render-thread local state for multi-pass sub-frame loops inside sub_822F16D0.
thread_local bool t_onRenderThread = false;
bool g_inSubFrameLoop = false;
SubFrameSnapshot g_snapshot{};
std::chrono::steady_clock::time_point g_subFrameStart{};

// VFR 60 Hz / 30 Hz logic window scheduler state (render-thread owned).
bool g_vfrWindowInit = false;
bool g_window30Fps = false;
bool g_passDecisionLatched = false;
bool g_passWillLoop = false;
bool g_stateBridgeDirty = false;
float g_maxPassAlpha = 0.0f;
int64_t g_windowStartNs = 0;
int64_t g_windowEndNs = 0;
int64_t g_windowPeriodNs = 16'666'667;
int64_t g_prevSubFrameNs = 0;

// 60 Hz logic vblank translation state (called from DispatchInterrupt).
uint32_t g_lastDevVblank = 0;
uint32_t g_logicVblank = 0;
uint32_t g_vblankSubAccum = 0;
bool g_vblankInit = false;

void ResetVfrWindow() {
  g_vfrWindowInit = false;
  g_window30Fps = false;
  g_passDecisionLatched = false;
  g_passWillLoop = false;
  g_maxPassAlpha = 0.0f;
  g_windowStartNs = 0;
  g_windowEndNs = 0;
  g_windowPeriodNs = 16'666'667;
  g_prevSubFrameNs = 0;
}

void CaptureSnapshot() {
  using ng2::guest::Load32;
  g_snapshot.postPasses = Load32(kPostPasses);
  for (uint32_t i = 0; i < 8; i++) {
    g_snapshot.fadeWords[i] = Load32(kFadeBase + i * 4);
  }
  g_snapshot.fadeResetFlag = Load32(kFadeResetFlag);
  g_snapshot.sceneActive = Load32(kSceneActive);
  g_snapshot.renderMgrSlotParity = Load32(kRenderMgrSlotParity);
  g_snapshot.stagePoseParity = Load32(kStagePoseParity);
  g_snapshot.renderCounter = Load32(kRenderCounter);
  g_snapshot.renderEndFlagA = Load32(kRenderEndFlagA);
  g_snapshot.renderEndFlagB = Load32(kRenderEndFlagB);
  g_snapshot.swapCounter = Load32(kSwapCounter);
  g_snapshot.vblankInterval = Load32(kVblankInterval);
  g_snapshot.gammaDirty = Load32(kGammaDirty);
  g_snapshot.tileResetFlag = Load32(kTileResetFlag);
  g_snapshot.ring0Addr = Load32(kRenderMgr + 1296);
  g_snapshot.ring1Addr = Load32(kRenderMgr + 1300);
  g_snapshot.ring0Count = g_snapshot.ring0Addr ? Load64(g_snapshot.ring0Addr) : 0;
  g_snapshot.ring1Count = g_snapshot.ring1Addr ? Load64(g_snapshot.ring1Addr) : 0;
}

void RestoreSnapshot() {
  Store32(kPostPasses, g_snapshot.postPasses);
  for (uint32_t i = 0; i < 8; i++) {
    Store32(kFadeBase + i * 4, g_snapshot.fadeWords[i]);
  }
  Store32(kFadeResetFlag, g_snapshot.fadeResetFlag);
  Store32(kSceneActive, g_snapshot.sceneActive);
  Store32(kRenderMgrSlotParity, g_snapshot.renderMgrSlotParity);
  Store32(kStagePoseParity, g_snapshot.stagePoseParity);
  Store32(kRenderCounter, g_snapshot.renderCounter);
  Store32(kRenderEndFlagA, g_snapshot.renderEndFlagA);
  Store32(kRenderEndFlagB, g_snapshot.renderEndFlagB);
  Store32(kSwapCounter, g_snapshot.swapCounter);
  Store32(kVblankInterval, g_snapshot.vblankInterval);
  Store32(kGammaDirty, g_snapshot.gammaDirty);
  Store32(kTileResetFlag, g_snapshot.tileResetFlag);
  if (g_snapshot.ring0Addr) {
    Store64(g_snapshot.ring0Addr, g_snapshot.ring0Count);
  }
  if (g_snapshot.ring1Addr) {
    Store64(g_snapshot.ring1Addr, g_snapshot.ring1Count);
  }
}

}  // namespace

RateMode CurrentRateMode() {
  if (!ng2::video::IsNativeRenderer()) {
    return RateMode::kOff60;
  }
  if (const char* v = std::getenv("NG2_VFR"); v && v[0]) {
    if (v[0] == '1') {
      return RateMode::kVariable;
    }
    if (v[0] == '0') {
      return RateMode::kOff60;
    }
  }
  if (const char* v = std::getenv("NG2_NATIVE_120FPS"); v && v[0]) {
    return v[0] == '1' ? RateMode::kFixed120 : RateMode::kOff60;
  }
  if (rex::cvar::GetFlagSource("frame_rate_mode") == rex::cvar::Source::kDefault &&
      REXCVAR_GET(native_120fps)) {
    return RateMode::kFixed120;
  }
  int32_t mode = std::clamp(REXCVAR_GET(frame_rate_mode), 0, 2);
  return RateMode(mode);
}

bool Enabled() {
  return CurrentRateMode() != RateMode::kOff60;
}

bool IsVfrActive() {
  return CurrentRateMode() == RateMode::kVariable;
}

RateMode ActiveFrameMode() {
  return RateMode(g_frameMode.load(std::memory_order_acquire));
}

bool IsActiveFrameVfr() {
  return ActiveFrameMode() == RateMode::kVariable;
}

bool IsRenderThreadFirstPass() {
  return t_onRenderThread && (IsFirstSubFrame() || CurrentPhase() == Phase::kDisabled);
}

void SetVblank120Active(bool active) {
  bool prev = g_vblank120Active.exchange(active, std::memory_order_release);
  if (prev != active) {
    REXLOG_INFO("high_fps: 120 Hz VBlank pacing {}", active ? "active (60 Hz logic)" : "inactive (60 Hz)");
  }
}

bool Vblank120Active() {
  return g_vblank120Active.load(std::memory_order_acquire);
}

bool IsActive() {
  RateMode mode = CurrentRateMode();
  if (mode == RateMode::kVariable) {
    return true;
  }
  if (mode == RateMode::kFixed120) {
    return Vblank120Active();
  }
  return false;
}

bool OverloadDegraded() {
  return g_overloadDegraded.load(std::memory_order_acquire);
}

bool IsOverloadDegraded() {
  return CurrentRateMode() == RateMode::kFixed120 && OverloadDegraded();
}

Phase CurrentPhase() {
  return Phase(g_phase.load(std::memory_order_acquire));
}

bool IsSubFrame0() {
  return IsFirstSubFrame();
}

bool IsSubFrame1() {
  Phase p = CurrentPhase();
  return p == Phase::kSubFrame1;
}

uint32_t SubFrameIndex() {
  return g_subFrameIndex.load(std::memory_order_acquire);
}

uint32_t SubFrameCount() {
  return g_subFrameCount.load(std::memory_order_acquire);
}

bool IsFirstSubFrame() {
  return CurrentPhase() != Phase::kDisabled && SubFrameIndex() == 0;
}

bool IsLastSubFrame() {
  Phase p = CurrentPhase();
  if (p == Phase::kDisabled) {
    return false;
  }
  if (IsActiveFrameVfr()) {
    return g_passDecisionLatched && !g_passWillLoop;
  }
  uint32_t idx = SubFrameIndex();
  uint32_t count = SubFrameCount();
  return count > 0 && idx + 1 >= count;
}

bool IsIntermediateSubFrame() {
  Phase p = CurrentPhase();
  if (p == Phase::kDisabled) {
    return false;
  }
  if (IsActiveFrameVfr()) {
    if (g_passDecisionLatched) {
      return g_passWillLoop;
    }
    // Before OnSwap latches the final window check, treat sub-frames with
    // alpha < 0.999 as intermediate so Phase 1 interpolation stays active.
    return SubFrameAlpha() < 0.999f;
  }
  uint32_t idx = SubFrameIndex();
  uint32_t count = SubFrameCount();
  return idx + 1 < count;
}

bool Is30FpsSubFrameMode() {
  return CurrentPhase() != Phase::kDisabled && g_is30FpsMode.load(std::memory_order_acquire);
}

uint64_t LogicFrameId() {
  return g_logicFrameId.load(std::memory_order_acquire);
}

float SubFrameAlpha() {
  if (CurrentPhase() == Phase::kDisabled) {
    return 1.0f;
  }
  return g_subFrameAlpha.load(std::memory_order_acquire);
}

double SubFrameDeltaSec() {
  return g_subFrameDeltaSec.load(std::memory_order_acquire);
}

double EffectiveVfrFpsLimit() {
  int32_t limit = REXCVAR_GET(vfr_fps_limit);
  if (const char* v = std::getenv("NG2_VFR_CAP"); v && v[0]) {
    limit = std::atoi(v);
  }
  if (limit < 0) {
    double hz = ng2::video::DisplayModeHz();
    if (hz <= 0.0) {
      if (int64_t periodNs = ng2::video::ModeRefreshNs(); periodNs > 0) {
        hz = 1e9 / double(periodNs);
      }
    }
    if (hz > 63.0) {
      return std::max(60.0, std::round(hz) - 3.0);
    }
    return 0.0;
  }
  return limit > 0 ? double(limit) : 0.0;
}

bool PrepareVfrSwapContinue() {
  if (!t_onRenderThread || CurrentPhase() == Phase::kDisabled || !IsActiveFrameVfr()) {
    return false;
  }
  if (SubFrameIndex() == 0 && !g_window30Fps && ShouldUseR4For30Fps()) {
    g_window30Fps = true;
    g_is30FpsMode.store(true, std::memory_order_release);
    g_windowPeriodNs = 33'333'333LL;
    g_windowEndNs = g_windowStartNs + g_windowPeriodNs;
    g_subFrameCount.store(4u, std::memory_order_release);
    g_snapshot.vblankInterval = ng2::guest::Load32(kVblankInterval);
    if (!g_lastLogged30Fps) {
      g_lastLogged30Fps = true;
      REXLOG_INFO("high_fps: promoted VFR window -> 30fps mid-frame (0x84C5342C={})",
                  g_snapshot.vblankInterval);
    }
  }
  return (SteadyNs() < g_windowEndNs) && (ng2::guest::Load32(kQuitFlag) == 0);
}

bool LatchVfrContinueAfterSwap(bool mightLoop, int64_t queuedSwapWaitNs, bool fifoPresent) {
  if (!t_onRenderThread || CurrentPhase() == Phase::kDisabled || !IsActiveFrameVfr()) {
    g_passWillLoop = false;
    g_passDecisionLatched = true;
    return false;
  }
  int64_t marginNs = 0;
  if (queuedSwapWaitNs > 0) {
    // Reserve the full queued Swap period when FIFO VSync blocks or frame
    // time approaches the 16.67 ms (60 Hz) logic cliff. In non-FIFO heavy
    // combat (~8-12 ms), render and DrawWorker/GPU overlap in parallel, so
    // reserving half avoids snapping from ~100 FPS down to 60 FPS.
    if (fifoPresent || queuedSwapWaitNs >= 13'000'000LL) {
      marginNs = queuedSwapWaitNs + 300'000LL;
    } else {
      marginNs = queuedSwapWaitNs / 2;
    }
  }
  const bool loop = mightLoop && (SteadyNs() + marginNs < g_windowEndNs) &&
                    (ng2::guest::Load32(kQuitFlag) == 0);
  g_passWillLoop = loop;
  g_passDecisionLatched = true;
  return loop;
}

bool IsPassDecisionLatched() {
  return g_passDecisionLatched;
}

bool DecideSubFrameContinue() {
  if (!t_onRenderThread || CurrentPhase() == Phase::kDisabled) {
    return false;
  }
  if (!IsActiveFrameVfr()) {
    return IsIntermediateSubFrame();
  }
  return LatchVfrContinueAfterSwap(PrepareVfrSwapContinue());
}

bool PassWillLoop() {
  return g_passDecisionLatched && g_passWillLoop;
}

void OnRenderSubFrameBegin() {
  t_onRenderThread = true;
  const RateMode mode = CurrentRateMode();
  if (!g_rateModeLogged || mode != g_lastLoggedRateMode) {
    g_rateModeLogged = true;
    g_lastLoggedRateMode = mode;
    REXLOG_INFO("high_fps: frame rate mode -> {}",
                mode == RateMode::kVariable ? "Variable (VFR)"
                : mode == RateMode::kFixed120 ? "120 FPS (Fixed)"
                                              : "60 FPS");
  }

  if (!g_inSubFrameLoop) {
    g_logicFrameId.fetch_add(1, std::memory_order_release);
    ng2::video::OnLogicFrameBegin();
    if (mode == RateMode::kVariable && ng2::guest::Load32(kQuitFlag) == 0) {
      g_frameMode.store(int(RateMode::kVariable), std::memory_order_release);
      ResetOverloadGuard();
      int64_t nowNs = SteadyNs();
      const bool use30Fps = ShouldUseR4For30Fps();
      const int64_t periodNs = use30Fps ? 33'333'333LL : 16'666'667LL;
      if (!g_vfrWindowInit || nowNs - g_windowEndNs > 2 * periodNs) {
        g_windowStartNs = nowNs;
        g_windowEndNs = nowNs + periodNs;
        g_vfrWindowInit = true;
      } else {
        g_windowEndNs = g_windowStartNs + periodNs;
        // Guardrail: never let logic exceed 60.000 TPS (or 30.000 TPS).
        if (nowNs < g_windowStartNs) {
          VfrSleepUntilNs(g_windowStartNs);
          nowNs = SteadyNs();
        }
      }
      g_windowPeriodNs = periodNs;
      g_window30Fps = use30Fps;
      g_is30FpsMode.store(use30Fps, std::memory_order_release);
      if (use30Fps != g_lastLogged30Fps) {
        g_lastLogged30Fps = use30Fps;
        REXLOG_INFO("high_fps: VFR logic window -> {} (0x84C5342C={}, 0x84C555D8={}, movie={})",
                    use30Fps ? "30 TPS UI/cutscene" : "60 TPS gameplay",
                    ng2::guest::Load32(kVblankInterval),
                    ng2::guest::Load32(kCutscene30fpsFlag),
                    ng2::movie::AnyFullscreenPlaying());
      }

      const double dtSec = (g_prevSubFrameNs > 0)
                               ? std::clamp(double(nowNs - g_prevSubFrameNs) * 1e-9, 1.0 / 1000.0, 0.05)
                               : (1.0 / 120.0);
      g_prevSubFrameNs = nowNs;
      const float alpha = float(std::clamp(
          double(nowNs - g_windowStartNs) / double(g_windowPeriodNs), 0.0, 1.0));
      g_maxPassAlpha = alpha;
      g_stateBridgeDirty = true;
      g_subFrameAlpha.store(alpha, std::memory_order_release);
      g_subFrameDeltaSec.store(dtSec, std::memory_order_release);
      g_subFrameCount.store(use30Fps ? 4u : 2u, std::memory_order_release);
      g_subFrameIndex.store(0, std::memory_order_release);
      g_passDecisionLatched = false;
      g_passWillLoop = false;
      CaptureSnapshot();
      g_inSubFrameLoop = true;
      g_phase.store(int(Phase::kSubFrame0), std::memory_order_release);
      input::PollSubFrame(true);
      camera::ApplySubFrameCamera(true, alpha, float(dtSec));
    } else if (mode == RateMode::kFixed120 && IsActive() && ng2::guest::Load32(kQuitFlag) == 0) {
      g_frameMode.store(int(RateMode::kFixed120), std::memory_order_release);
      ResetVfrWindow();
      g_stateBridgeDirty = true;
      g_subFrameStart = std::chrono::steady_clock::now();
      g_logicRenderTotalNs = 0;
      g_logicMaxSubFrameNs = 0;
      const bool useR4 = ShouldUseR4For30Fps();
      g_is30FpsMode.store(useR4, std::memory_order_release);
      if (useR4 != g_lastLogged30Fps) {
        g_lastLogged30Fps = useR4;
        REXLOG_INFO("high_fps: sub-frame mode -> {} (R={}, 0x84C5342C={}, 0x84C555D8={}, movie={})",
                    useR4 ? "30fps UI/cutscene -> 120fps" : "60fps gameplay -> 120fps",
                    useR4 ? 4 : 2, ng2::guest::Load32(kVblankInterval),
                    ng2::guest::Load32(kCutscene30fpsFlag), ng2::movie::AnyFullscreenPlaying());
      }
      const uint32_t count = useR4 ? 4u : 2u;
      const float alpha = 1.0f / float(count);
      g_subFrameAlpha.store(alpha, std::memory_order_release);
      g_subFrameDeltaSec.store(1.0 / 120.0, std::memory_order_release);
      g_subFrameCount.store(count, std::memory_order_release);
      g_subFrameIndex.store(0, std::memory_order_release);
      g_passDecisionLatched = false;
      g_passWillLoop = false;
      CaptureSnapshot();
      g_inSubFrameLoop = true;
      g_phase.store(int(Phase::kSubFrame0), std::memory_order_release);
      input::PollSubFrame(true);
      camera::ApplySubFrameCamera(true, alpha, 1.0f / 120.0f);
    } else {
      const bool trackDegraded =
          (mode == RateMode::kFixed120 &&
           g_overloadDegraded.load(std::memory_order_acquire) &&
           ng2::guest::Load32(kQuitFlag) == 0);
      g_frameMode.store(int(trackDegraded ? RateMode::kFixed120 : RateMode::kOff60),
                        std::memory_order_release);
      ResetVfrWindow();
      if (g_stateBridgeDirty) {
        g_stateBridgeDirty = false;
        state_bridge::Reset();
      }
      if (!Enabled()) {
        ResetOverloadGuard();
      }
      g_subFrameStart = trackDegraded ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
      g_inSubFrameLoop = false;
      g_is30FpsMode.store(false, std::memory_order_release);
      g_subFrameAlpha.store(1.0f, std::memory_order_release);
      g_subFrameDeltaSec.store(1.0 / 60.0, std::memory_order_release);
      g_subFrameIndex.store(0, std::memory_order_release);
      g_subFrameCount.store(1, std::memory_order_release);
      g_phase.store(int(Phase::kDisabled), std::memory_order_release);
    }
  } else if (ActiveFrameMode() == RateMode::kVariable) {
    const int64_t nowNs = SteadyNs();
    const double dtSec = (g_prevSubFrameNs > 0)
                             ? std::clamp(double(nowNs - g_prevSubFrameNs) * 1e-9, 1.0 / 1000.0, 0.05)
                             : (1.0 / 120.0);
    g_prevSubFrameNs = nowNs;
    float alpha = float(std::clamp(
        double(nowNs - g_windowStartNs) / double(g_windowPeriodNs), 0.0, 1.0));
    alpha = std::max(alpha, g_maxPassAlpha);
    g_maxPassAlpha = alpha;
    g_subFrameAlpha.store(alpha, std::memory_order_release);
    g_subFrameDeltaSec.store(dtSec, std::memory_order_release);
    g_subFrameIndex.fetch_add(1, std::memory_order_release);
    g_passDecisionLatched = false;
    g_passWillLoop = false;
    g_phase.store(int(Phase::kSubFrame1), std::memory_order_release);
    input::PollSubFrame(false);
    camera::ApplySubFrameCamera(false, alpha, float(dtSec));
  } else {
    g_subFrameStart = std::chrono::steady_clock::now();
    const uint32_t count = g_subFrameCount.load(std::memory_order_acquire);
    const uint32_t idx = g_subFrameIndex.load(std::memory_order_acquire) + 1;
    const float alpha = float(idx + 1) / float(count);
    g_subFrameAlpha.store(alpha, std::memory_order_release);
    g_subFrameDeltaSec.store(1.0 / 120.0, std::memory_order_release);
    g_subFrameIndex.store(idx, std::memory_order_release);
    g_inSubFrameLoop = (idx + 1 < count);
    g_phase.store(int(Phase::kSubFrame1), std::memory_order_release);
    input::PollSubFrame(false);
    camera::ApplySubFrameCamera(false, alpha, 1.0f / 120.0f);
  }
}

bool ShouldLoopRenderSubFrame() {
  Phase phase = Phase(g_phase.load(std::memory_order_acquire));
  if (IsActiveFrameVfr() && (phase == Phase::kSubFrame0 || phase == Phase::kSubFrame1)) {
    if (!g_passDecisionLatched) {
      DecideSubFrameContinue();
    }
    const bool quit = ng2::guest::Load32(kQuitFlag) != 0;
    if (g_passWillLoop && !quit) {
      g_inSubFrameLoop = true;
      RestoreSnapshot();
      return true;
    }
    const int64_t nowExitNs = SteadyNs();
    if (quit) {
      ResetVfrWindow();
    } else if (nowExitNs - g_windowEndNs > g_windowPeriodNs) {
      g_windowStartNs = nowExitNs;
    } else {
      g_windowStartNs = g_windowEndNs;
    }
    camera::RestoreLogicCamera();
    g_inSubFrameLoop = false;
    g_passDecisionLatched = false;
    g_passWillLoop = false;
    g_maxPassAlpha = 0.0f;
    g_is30FpsMode.store(false, std::memory_order_release);
    g_subFrameAlpha.store(1.0f, std::memory_order_release);
    g_subFrameIndex.store(0, std::memory_order_release);
    g_subFrameCount.store(1, std::memory_order_release);
    g_phase.store(int(Phase::kDisabled), std::memory_order_release);
    g_frameMode.store(int(RateMode::kOff60), std::memory_order_release);
    return false;
  }

  if (phase == Phase::kSubFrame0 || phase == Phase::kSubFrame1) {
    int64_t ns = 0;
    if (g_subFrameStart != std::chrono::steady_clock::time_point()) {
      ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now() - g_subFrameStart)
               .count();
      NoteSubFrameDurationNs(ns);
      g_logicRenderTotalNs += ns;
      if (ns > g_logicMaxSubFrameNs) {
        g_logicMaxSubFrameNs = ns;
      }
    }
    uint32_t idx = g_subFrameIndex.load(std::memory_order_acquire);
    uint32_t count = g_subFrameCount.load(std::memory_order_acquire);
    // If sub_8363A6A8 turned on 0x84C5342C=2 mid-SF0 (entering a 30fps cutscene
    // without a movie), promote this pass to R=4 and retain vblankInterval=2.
    if (idx == 0 && count == 2 && ShouldUseR4For30Fps()) {
      count = 4;
      g_is30FpsMode.store(true, std::memory_order_release);
      g_subFrameCount.store(count, std::memory_order_release);
      g_snapshot.vblankInterval = ng2::guest::Load32(kVblankInterval);
      if (!g_lastLogged30Fps) {
        g_lastLogged30Fps = true;
        REXLOG_INFO("high_fps: promoted sub-frame mode -> R=4 mid-frame (0x84C5342C={})",
                    g_snapshot.vblankInterval);
      }
    }
    if (idx + 1 < count) {
      if (ng2::guest::Load32(kQuitFlag) != 0) {
        camera::RestoreLogicCamera();
        g_inSubFrameLoop = false;
        g_is30FpsMode.store(false, std::memory_order_release);
        g_subFrameAlpha.store(1.0f, std::memory_order_release);
        g_subFrameIndex.store(0, std::memory_order_release);
        g_subFrameCount.store(1, std::memory_order_release);
        g_phase.store(int(Phase::kDisabled), std::memory_order_release);
        g_frameMode.store(int(RateMode::kOff60), std::memory_order_release);
        return false;
      }
      g_inSubFrameLoop = true;
      RestoreSnapshot();
      return true;
    }
    EvaluateLogicFrameLoad(g_logicRenderTotalNs, g_logicMaxSubFrameNs, count);
  } else if (ActiveFrameMode() == RateMode::kFixed120 &&
             g_overloadDegraded.load(std::memory_order_acquire) &&
             g_subFrameStart != std::chrono::steady_clock::time_point()) {
    int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - g_subFrameStart)
                     .count();
    EvaluateLogicFrameLoad(ns, ns, 1);
  }
  camera::RestoreLogicCamera();
  g_inSubFrameLoop = false;
  g_is30FpsMode.store(false, std::memory_order_release);
  g_subFrameAlpha.store(1.0f, std::memory_order_release);
  g_subFrameIndex.store(0, std::memory_order_release);
  g_subFrameCount.store(1, std::memory_order_release);
  g_phase.store(int(Phase::kDisabled), std::memory_order_release);
  g_frameMode.store(int(RateMode::kOff60), std::memory_order_release);
  return false;
}

uint32_t TranslateLogicVblank(uint32_t devVblankCount) {
  if (!g_vblankInit) {
    g_vblankInit = true;
    g_lastDevVblank = devVblankCount;
    g_logicVblank = devVblankCount;
    g_vblankSubAccum = 0;
    return g_logicVblank;
  }
  uint32_t delta = devVblankCount - g_lastDevVblank;
  g_lastDevVblank = devVblankCount;
  if (!IsVfrActive() && Vblank120Active()) {
    g_vblankSubAccum += delta;
    g_logicVblank += g_vblankSubAccum / 2;
    g_vblankSubAccum %= 2;
  } else {
    g_logicVblank += delta;
    g_vblankSubAccum = 0;
  }
  return g_logicVblank;
}

void NoteSubFrameDurationNs(int64_t ns) {
  if (!IsActive() || IsVfrActive()) {
    return;
  }
  static uint32_t slowStreak = 0;
  static uint32_t slowWarnings = 0;
  // Normal 120 Hz VSync-locked sub-frames sit at ~8.33 ms wall time; only warn
  // when sustained sub-frame duration actually falls behind 120 Hz (> 10.0 ms).
  if (ns > 10'000'000) {
    if (++slowStreak == 32 && slowWarnings < 10) {
      slowWarnings++;
      REXLOG_WARN("high_fps: 32 consecutive sub-frames exceeded 10.0 ms (last {:.2f} ms)",
                  double(ns) / 1e6);
    }
  } else {
    slowStreak = 0;
  }
}

}  // namespace ng2::high_fps
