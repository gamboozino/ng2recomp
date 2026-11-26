// ng2 - Ninja Gaiden II static recompilation
//
// Native 120 FPS decoupled rendering (60 Hz deterministic game logic +
// 120 Hz render thread sub-frames). Phase 1: base state machine, VBlank
// separation, and render-thread double-pass side-effect isolation.

#pragma once

#include <cstdint>

namespace ng2::high_fps {

enum class RateMode : int32_t {
  kOff60 = 0,     // Original 60 FPS
  kFixed120 = 1,  // Fixed 120 FPS (compatibility / reference mode)
  kVariable = 2,  // Native Variable Frame Rate (60~240+ FPS / 144Hz / VRR)
};

enum class Phase { kDisabled, kSubFrame0, kSubFrame1 };

// Active frame rate mode resolved from frame_rate_mode / native_120fps / env.
RateMode CurrentRateMode();
// Latched frame rate mode for the currently executing logic frame on the render thread.
RateMode ActiveFrameMode();
bool IsActiveFrameVfr();
bool IsRenderThreadFirstPass();

// True when native 120 FPS or VFR is enabled and the native Vulkan renderer is active.
bool Enabled();

// True when RateMode::kVariable (VFR) is active on the native Vulkan renderer.
bool IsVfrActive();

// True when 120 Hz VBlank pacing is active on the display/present side.
void SetVblank120Active(bool active);
bool Vblank120Active();

// True when decoupled high-FPS rendering is actively running (VFR active, or
// fixed 120 Hz with Vblank120Active()).
bool IsActive();

// Current sub-frame phase on the render thread.
Phase CurrentPhase();
bool IsSubFrame0();
bool IsSubFrame1();

// Sub-frame index & multiplier within the current logic frame:
// Fixed 120: R = 2 (alphas 0.50, 1.00) for 60 FPS gameplay,
//            R = 4 (alphas 0.25, 0.50, 0.75, 1.00) for 30 FPS UI / cutscenes.
// VFR: dynamic N passes within the 60 Hz (or 30 Hz) logic window.
uint32_t SubFrameIndex();
uint32_t SubFrameCount();
bool IsFirstSubFrame();
bool IsLastSubFrame();
bool IsIntermediateSubFrame();
bool Is30FpsSubFrameMode();

// Monotonic 60 Hz logic frame generation id (incremented once per logic frame).
uint64_t LogicFrameId();

// Continuous interpolation fraction within the current logic frame:
// Fixed 120: (idx + 1) / count; VFR: clamp((t_now - T_start) / T_period, 0, 1).
float SubFrameAlpha();

// Real elapsed time (seconds) between the previous sub-frame and current sub-frame.
double SubFrameDeltaSec();

// Effective host FPS cap in VFR mode (0.0 = unlimited / VSync paced).
double EffectiveVfrFpsLimit();

// Called at OnSwap on the render thread to atomically latch whether this
// pass loops back for another sub-frame or concludes the current logic frame.
bool DecideSubFrameContinue();

// Two-stage VFR loop decision around OnSwap / WaitForWorker:
// PrepareVfrSwapContinue() handles mid-frame 30fps promotion and checks if the
// logic window has time left before WaitForWorker.
// LatchVfrContinueAfterSwap() latches the authoritative loop decision after
// WaitForWorker / RunSwap returns, reserving queuedSwapWaitNs for a still-
// queued worker swap so VSync FIFO and GPU bottlenecks never over-queue passes.
bool PrepareVfrSwapContinue();
bool LatchVfrContinueAfterSwap(bool mightLoop, int64_t queuedSwapWaitNs = 0,
                               bool fifoPresent = false);
bool IsPassDecisionLatched();

// Returns the latched loop decision from DecideSubFrameContinue() /
// LatchVfrContinueAfterSwap().
bool PassWillLoop();

// Called at loc_822F1880 (start of sub_822F16D0 frame body) and at 0x822F21B8
// (tail before clearing 0x84C53430). ShouldLoopRenderSubFrame() restores
// single-render side-effect state and returns true to jump back to 0x822F1880.
void OnRenderSubFrameBegin();
bool ShouldLoopRenderSubFrame();

// Accumulates 120 Hz D3D vblank ticks into a 60 Hz logic vblank counter for
// 0x84C55B50 so game timers and frame limiters remain strictly at 60 Hz.
uint32_t TranslateLogicVblank(uint32_t devVblankCount);

// True when dynamic overload protection has temporarily degraded R -> 1 (60 Hz)
// to protect 60 Hz game logic pacing.
bool IsOverloadDegraded();
// Raw overload flag (independent of Vblank120Active) so VsyncThread can drop
// physical vblank pacing back to 60 Hz in lockstep with R = 1.
bool OverloadDegraded();

// Records sub-frame render+present CPU/GPU wall duration for overload
// monitoring and dynamic R=2 <-> R=1 protection.
void NoteSubFrameDurationNs(int64_t ns);

}  // namespace ng2::high_fps
