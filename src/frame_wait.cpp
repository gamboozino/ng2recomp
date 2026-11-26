// ng2 - Ninja Gaiden II static recompilation
//
// Three of the game's busy-waits, which on the 360 had hardware threads to
// themselves and here hold whole cores: on a 4-core PC they take CPU time
// from the threads they wait for.
//
// The main loop (sub_822F34C0) hands each frame to the render thread, then
// spins until the render thread has taken it (lwz r11 of 0x84C53430 until
// it reads 0): nearly all of the main thread's time on a fast PC.
//
// The render thread (sub_822F16D0) spins the other way, until the main loop
// has handed it the next frame (0x84C53430, then 0x84C53434, set): nearly a
// whole core whenever the main loop is the slower one, the usual case on a
// busy 4-core PC, where it starved the audio threads (#39).
//
// The render thread's frame limiter (sub_822F0BF8) spins until the vblank
// count (0x84C55B50, from the vblank interrupt) is the word at 0x84C5342C
// past where it started, with the PowerPC low-priority hint in the loop.
//
// When the game asks for fewer frames than vblanks (the word at 0x84C5342C:
// 2 on the game over screen), the limiter reads the count (sub_822F02D8)
// once to start, then until it has moved on by that many. On the console the
// frame's work before it fits in a vblank, so it starts in the vblank the
// last wait ended in and every frame takes exactly 2. Here Present can wait
// on the host display's vsync, which isn't in step with the guest's vblanks,
// and push the start a vblank or two late: that frame took 3 or 4 (27 fps,
// not 30, uneven). A start up to the interval past the last wait's end
// counts from that end, so a frame takes the interval unless its own work
// took longer than that (where the console would also run late).
//
// All three back off: a few pause instructions, then yields, then short sleeps (yields on
// Windows, whose sleeps are whole milliseconds).
//
// Also here, as it is the same handoff:
//
// The poses: the render thread starts each frame by working out where the
// stage's animated parts are (sub_8247EA10: sliding doors and the like), and
// with no scripted scene running it reads their poses (sub_83751980) from the
// array the game's update tasks work in, not from a copy kept for the frame.
// Each tick one task puts those poses back to rest and a later one animates
// them again. On the console the render thread reads them as soon as the
// main loop starts it on the frame (sub_822F0CA8), with the tick's tasks
// done. Here it is often still waiting on the display with the last frame
// when the next tick's tasks run, and then read a pose put back to rest and
// not yet animated: the left leaf of Muramasa's door in Chapter 1 drawn
// 11 cm off, on and off from frame to frame (#21). The poses the render
// thread read in its last frame are now copied as the main loop starts it on
// the next, and its reads of them that frame (any made on the render thread
// with no scene running) get the copies. A part read for the first time still
// reads the array for a frame or two. NG2_POSE_SNAPSHOT=0 has it read the
// array again.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <atomic>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include <rex/hook.h>

#include "gpu/guest.h"
#include "gpu/video.h"
#include "gpu/win_display.h"
#include "high_fps/camera_120hz.h"
#include "high_fps/high_fps.h"
#include "high_fps/state_bridge.h"

namespace {

// Time spent in each kind of wait (NG2_PROFILE: ng2::video::TakeWaitTimes).
std::atomic<int64_t> g_waitNs[int(ng2::video::WaitKind::kCount)];

// One step of a spin; a new wait (a gap since the last step) starts over.
void Backoff(ng2::video::WaitKind kind) {
  using clock = std::chrono::steady_clock;
  thread_local uint32_t spins = 0;
  thread_local clock::time_point last;
  auto now = clock::now();
  if (now - last > std::chrono::microseconds(500)) {
    spins = 0;
  } else {
    g_waitNs[int(kind)].fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - last).count(), std::memory_order_relaxed);
  }
  spins++;
  if (spins < 64) {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#endif
  } else if (spins < 256) {
    std::this_thread::yield();
  } else {
#if defined(_WIN32)
    // MSVC's sleep_for rounds up to whole milliseconds (Sleep(1), 1-2 ms): too coarse here.
    // The high-resolution timer instead; a yield alone (Windows before 10 1803)
    // returns at once with a core free, keeping this thread busy (#17).
    if (ng2::video::PreciseSleepAvailable()) {
      ng2::video::PreciseSleepNs(50'000);
    } else {
      std::this_thread::yield();
    }
#else
    std::this_thread::sleep_for(std::chrono::microseconds(50));
#endif
  }
  last = clock::now();
}

}  // namespace

void ng2::video::NoteWait(WaitKind kind, int64_t ns) {
  g_waitNs[int(kind)].fetch_add(ns, std::memory_order_relaxed);
}

void ng2::video::TakeWaitTimes(int64_t* ns) {
  for (int k = 0; k < int(WaitKind::kCount); k++) {
    ns[k] = g_waitNs[k].exchange(0, std::memory_order_relaxed);
  }
}

// Before the compare: r11 is the handoff word, 0 once the frame is taken.
void Ng2FrameHandoffWait(PPCRegister& r11) {
  if (r11.u32 != 0) {
    Backoff(ng2::video::WaitKind::kHandoff);
  } else {
    ng2::video::NoteFrameTaken();  // (NG2_LATENCY_LOG)
  }
}

// At each loop's quit-flag check, reached only while the word is still 0.
void Ng2FrameWait() {
  Backoff(ng2::video::WaitKind::kFrame);
}

namespace {
constexpr uint32_t kBufferIndex = 0x84C3DA38;  // 0 or 1: the frame's side of what is kept twice
constexpr uint32_t kPoseOffsets = 0x83AA56C8 + 704;  // + buffer * 4: the pose array's place in an object
constexpr uint32_t kStaticBegin = 0x82000000, kStaticEnd = 0x85A90000;  // the game's image (always there)

struct PoseCopy {
  uint32_t address;
  uint8_t data[64];
};

bool PoseSnapshotOn() {
  static const bool on = [] {
    const char* v = std::getenv("NG2_POSE_SNAPSHOT");
    return !(v && v[0] == '0');
  }();
  return on;
}

// Between the two threads (held for a copy of a few kilobytes at most).
std::mutex g_poseMutex;
std::vector<uint32_t> g_poseWanted;     // the poses the render thread read last frame
std::vector<PoseCopy> g_posePending;    // as they were when the main loop last started a frame
bool g_posePendingNew = false;
// The render thread's own.
thread_local bool t_renderThread = false;
thread_local std::vector<PoseCopy> t_poses;  // by address
thread_local std::vector<uint32_t> t_poseRead;
}  // namespace

// The main loop, about to start the render thread on a frame.
REX_EXTERN(__imp__sub_822F0CA8);
REX_HOOK_RAW(sub_822F0CA8) {
  if (ng2::high_fps::IsActive()) {
    ng2::high_fps::state_bridge::OnLogicHandoff();
    ng2::high_fps::camera::OnLogicHandoff();
  } else if (PoseSnapshotOn()) {
    std::lock_guard lock(g_poseMutex);
    g_posePending.resize(g_poseWanted.size());
    for (size_t i = 0; i < g_poseWanted.size(); i++) {
      g_posePending[i].address = g_poseWanted[i];
      std::memcpy(g_posePending[i].data, ng2::guest::Virtual(g_poseWanted[i]), 64);
    }
    g_posePendingNew = true;
  }
  __imp__sub_822F0CA8(ctx, base);
}

// The render thread, about to place the stage's parts: advances Phase 3 S[n-1]/S[n]
// pose snapshots (or falls back to PoseSnapshotOn when 120fps is inactive).
REX_EXTERN(__imp__sub_8247EA10);
REX_HOOK_RAW(sub_8247EA10) {
  t_renderThread = true;
  if (ng2::high_fps::IsActive()) {
    ng2::high_fps::state_bridge::OnRenderFrameBegin();
  } else if (PoseSnapshotOn()) {
    std::sort(t_poseRead.begin(), t_poseRead.end());
    t_poseRead.erase(std::unique(t_poseRead.begin(), t_poseRead.end()), t_poseRead.end());
    {
      std::lock_guard lock(g_poseMutex);
      if (g_posePendingNew) {
        t_poses.swap(g_posePending);  // (sorted: made from a sorted list)
        g_posePendingNew = false;
      }
      g_poseWanted = t_poseRead;
    }
    t_poseRead.clear();
  }
  __imp__sub_8247EA10(ctx, base);
}

namespace {

template <typename ImpFn>
void RunPoseReader64(uint8_t readerKind, ImpFn imp, PPCContext& ctx, uint8_t* base) {
  const uint32_t outGuest = ctx.r3.u32;
  const uint32_t entryPtr = ctx.r4.u32;
  const uint32_t partIdx = (readerKind == 2) ? 0u : ctx.r5.u32;
  const uint32_t arg3 = (readerKind == 2) ? ctx.r5.u32 : ctx.r6.u32;

  if (t_renderThread && ng2::high_fps::IsActive()) {
    if (ng2::high_fps::state_bridge::TryReadInterpolatedPose64(
            readerKind, outGuest, entryPtr, partIdx, arg3)) {
      return;
    }
    imp(ctx, base);
    ng2::high_fps::state_bridge::RecordFallbackPose64(
        readerKind, entryPtr, partIdx, arg3, ng2::guest::Virtual(outGuest));
    return;
  }

  if (readerKind == 0 && t_renderThread && arg3 == 0 && PoseSnapshotOn()) {
    uint8_t buffer = *ng2::guest::Virtual(entryPtr + 17 + ng2::guest::Load32(kBufferIndex));
    if (buffer < 8) {  // (255: no pose, the original gives the identity)
      uint32_t address = ng2::guest::Load32(kPoseOffsets + buffer * 4) + partIdx * 64 +
                         ng2::guest::Load32(entryPtr);
      if (address >= kStaticBegin && address < kStaticEnd && kStaticEnd - address >= 64) {
        t_poseRead.push_back(address);
        auto found = std::lower_bound(t_poses.begin(), t_poses.end(), address,
                                      [](const PoseCopy& pose, uint32_t a) { return pose.address < a; });
        if (found != t_poses.end() && found->address == address) {
          std::memcpy(ng2::guest::Virtual(outGuest), found->data, 64);
          return;
        }
      }
    }
  }
  imp(ctx, base);
}

template <typename ImpFn>
void RunPartPoseBuilder(ImpFn imp, PPCContext& ctx, uint8_t* base) {
  const uint32_t outGuest = ctx.r3.u32;
  const uint32_t entryPtr = ctx.r4.u32;
  const uint32_t partIdx = ctx.r5.u32;
  const uint32_t arg3 = ctx.r6.u32;

  imp(ctx, base);
  const uint32_t resGuest = ctx.r3.u32;
  if (ng2::high_fps::IsActive() &&
      ng2::high_fps::state_bridge::ShouldCapturePureCurr(entryPtr) &&
      !ng2::high_fps::state_bridge::HasPureCurrForFrame(entryPtr, partIdx) &&
      resGuest != 0) {
    uint8_t interpSaved[64];
    std::memcpy(interpSaved, ng2::guest::Virtual(resGuest), 64);

    ctx.r3.u32 = outGuest;
    ctx.r4.u32 = entryPtr;
    ctx.r5.u32 = partIdx;
    ctx.r6.u32 = arg3;
    ng2::high_fps::state_bridge::SetForceCurrPose(true);
    imp(ctx, base);
    ng2::high_fps::state_bridge::SetForceCurrPose(false);
    const uint32_t currGuest = ctx.r3.u32 ? ctx.r3.u32 : outGuest;

    ng2::high_fps::state_bridge::RecordPureCurrPartMatrix(
        entryPtr, partIdx, ng2::guest::Virtual(currGuest));

    std::memcpy(ng2::guest::Virtual(resGuest), interpSaved, 64);
    ctx.r3.u64 = resGuest;
  }
}

template <typename ImpFn>
void RunBoneBuilder(ImpFn imp, PPCContext& ctx, uint8_t* base) {
  if (ng2::high_fps::IsActive()) {
    ng2::high_fps::state_bridge::BoneBuildScope scope(ctx.r3.u32);
    imp(ctx, base);
    return;
  }
  imp(ctx, base);
}

}  // namespace

// A part's pose (r3: 64 bytes out, r4: its object's entry, r5: the part, r6:
// which of the entry's buffer bytes).
REX_EXTERN(__imp__sub_83751980);
REX_HOOK_RAW(sub_83751980) {
  RunPoseReader64(0, __imp__sub_83751980, ctx, base);
}

REX_EXTERN(__imp__sub_83751AD0);
REX_HOOK_RAW(sub_83751AD0) {
  RunPoseReader64(1, __imp__sub_83751AD0, ctx, base);
}

REX_EXTERN(__imp__sub_83751E00);
REX_HOOK_RAW(sub_83751E00) {
  RunPoseReader64(2, __imp__sub_83751E00, ctx, base);
}

// Part pose builders writing 0x83AA56C8+736 and forwarding matrices to rigid
// draws; keep pure S[n] Curr roots for skeletal builders below.
REX_EXTERN(__imp__sub_8247DAC8);
REX_HOOK_RAW(sub_8247DAC8) {
  RunPartPoseBuilder(__imp__sub_8247DAC8, ctx, base);
}

REX_EXTERN(__imp__sub_8247FFB0);
REX_HOOK_RAW(sub_8247FFB0) {
  RunPartPoseBuilder(__imp__sub_8247FFB0, ctx, base);
}

// Skeletal matrix builders (sole callers of sub_8232EAA0).
REX_EXTERN(__imp__sub_82482D18);
REX_HOOK_RAW(sub_82482D18) {
  RunBoneBuilder(__imp__sub_82482D18, ctx, base);
}

REX_EXTERN(__imp__sub_82482F80);
REX_HOOK_RAW(sub_82482F80) {
  RunBoneBuilder(__imp__sub_82482F80, ctx, base);
}

// Vertex Fetch Slot 94 bone palette uploader.
REX_EXTERN(__imp__sub_8232EAA0);
REX_HOOK_RAW(sub_8232EAA0) {
  if (ng2::high_fps::state_bridge::HandleBonePaletteWrite(
          ctx.r3.u32, ctx.r4.u32, ctx.r5.u32, ctx.r6.u32)) {
    return;
  }
  __imp__sub_8232EAA0(ctx, base);
}

void Ng2VblankWait() {
  Backoff(ng2::video::WaitKind::kVblank);
}

// The limiter's reads of the vblank count: its start (returning to 822F0C40)
// and each check in its loop (to 822F0C4C).
REX_EXTERN(__imp__sub_822F02D8);
REX_HOOK_RAW(sub_822F02D8) {
  constexpr uint32_t kStart = 0x822F0C40, kCheck = 0x822F0C4C, kInterval = 0x84C5342C;
  static uint32_t lastEnd = 0;
  static bool waited = false;
  uint32_t lr = uint32_t(ctx.lr);
  __imp__sub_822F02D8(ctx, base);
  if (lr == kStart) {
    // Up to the interval since the last wait ended: blocked, not slow.
    uint32_t interval = ng2::guest::Load32(kInterval);
    if (waited && ctx.r3.u32 - lastEnd <= interval) {
      ctx.r3.u32 = lastEnd;
    }
  } else if (lr == kCheck) {
    lastEnd = ctx.r3.u32;  // the last check is the one the wait ended on
    waited = true;
  }
}

namespace {
thread_local bool t_pendingQueueDrain = false;
thread_local uint32_t t_pendingQueueDrainArg = 0;
}  // namespace

void Ng2RenderSubFrameBegin() {
  ng2::high_fps::OnRenderSubFrameBegin();
  if (ng2::high_fps::IsFirstSubFrame() ||
      ng2::high_fps::CurrentPhase() == ng2::high_fps::Phase::kDisabled) {
    t_pendingQueueDrain = false;
  }
}

bool Ng2RenderSubFrameEnd() {
  return ng2::high_fps::ShouldLoopRenderSubFrame();
}

// Render-thread drain of the 0x84E9C420 queue (sub_8233B618 at LR 0x822F1E8C):
// skip on intermediate sub-frames so the same logic queue stays intact until
// the final sub-frame of the logic frame. In VFR mode, delay execution until
// sub_822F0BF8 (after OnSwap latches PassWillLoop()) so every pass sees the
// full queue and it drains once at logic-frame exit.
REX_EXTERN(__imp__sub_8233B618);
REX_HOOK_RAW(sub_8233B618) {
  if (uint32_t(ctx.lr) == 0x822F1E8C) {
    if (ng2::high_fps::IsActiveFrameVfr()) {
      t_pendingQueueDrain = true;
      t_pendingQueueDrainArg = ctx.r3.u32;
      return;
    }
    if (ng2::high_fps::IsIntermediateSubFrame()) {
      return;
    }
  }
  __imp__sub_8233B618(ctx, base);
}

// 60 Hz frame-limiter interval update (sub_822F0BF8): do not clear 0x84C5342C
// prematurely on intermediate sub-frames. In VFR mode, the 33.33 ms logic
// window paces 30 fps sections directly without VBlank spinning, and any
// deferred 0x84E9C420 queue drain runs once on the final pass.
REX_EXTERN(__imp__sub_822F0BF8);
REX_HOOK_RAW(sub_822F0BF8) {
  auto drainPendingQueue = [&]() {
    if (t_pendingQueueDrain) {
      t_pendingQueueDrain = false;
      PPCContext saved = ctx;
      ctx.r3.u64 = t_pendingQueueDrainArg ? t_pendingQueueDrainArg : 0x84E9C420u;
      __imp__sub_8233B618(ctx, base);
      ctx = saved;
    }
  };
  if (ng2::high_fps::IsActiveFrameVfr()) {
    if (!ng2::high_fps::IsPassDecisionLatched()) {
      ng2::high_fps::DecideSubFrameContinue();
    }
    if (!ng2::high_fps::PassWillLoop()) {
      *reinterpret_cast<uint32_t*>(ng2::guest::Virtual(0x84C5342C)) = 0;
      drainPendingQueue();
    }
    return;
  }
  if (ng2::high_fps::IsIntermediateSubFrame()) {
    return;
  }
  drainPendingQueue();
  __imp__sub_822F0BF8(ctx, base);
}

// D3D vblank interrupt callback target sub_837EC940(r3 = &deviceVblankCount):
// translate 120 Hz physical D3D vblanks into a strict 60 Hz logic counter in
// 0x84C55B50 so game timers and frame limiters never double-tick.
REX_EXTERN(__imp__sub_837EC940);
REX_HOOK_RAW(sub_837EC940) {
  uint32_t ptr = ctx.r3.u32;
  uint32_t devCount = ng2::guest::Load32(ptr);
  uint32_t logicCount = ng2::high_fps::TranslateLogicVblank(devCount);
  auto* slot = reinterpret_cast<uint32_t*>(ng2::guest::Virtual(ptr));
  *slot = rex::byte_swap(logicCount);
  __imp__sub_837EC940(ctx, base);
  *slot = rex::byte_swap(devCount);
}

// Render-thread sync wait (sub_822F2420): under decoupled 120 Hz rendering,
// clear 0x84C53440 directly without blocking or stalling sub-frame pacing.
REX_EXTERN(__imp__sub_822F2420);
REX_HOOK_RAW(sub_822F2420) {
  if (ng2::high_fps::IsActive()) {
    *reinterpret_cast<uint32_t*>(ng2::guest::Virtual(0x84C53440)) = 0;
    ctx.r3.s64 = 0;
    return;
  }
  __imp__sub_822F2420(ctx, base);
}

// Render-thread UI fade reset (sub_82333C20): only clear fade counters and mark
// the active slot dirty without stepping animation timers.
REX_EXTERN(__imp__sub_82333C20);
REX_HOOK_RAW(sub_82333C20) {
  if (ng2::high_fps::IsActive()) {
    constexpr uint32_t kFadeStructBase = 0x83C10000;
    *reinterpret_cast<uint32_t*>(ng2::guest::Virtual(kFadeStructBase + 4228)) = 0;
    *reinterpret_cast<uint32_t*>(ng2::guest::Virtual(kFadeStructBase + 4232)) = 0;
    *reinterpret_cast<uint32_t*>(ng2::guest::Virtual(kFadeStructBase + 4360)) = 0;
    uint32_t slot = ng2::guest::Load32(kFadeStructBase + 4336);
    *reinterpret_cast<uint32_t*>(ng2::guest::Virtual(kFadeStructBase + 4352 + (slot << 2))) =
        rex::byte_swap(1u);
    return;
  }
  __imp__sub_82333C20(ctx, base);
}
