// ng2 - Ninja Gaiden II static recompilation
//
// Native 120 Hz camera stepping & projection synchronization for decoupled rendering:
// - Captures 60 Hz logical camera snapshots Cam[n-1] and Cam[n] at sub_822FA8C8.
// - Before Sub-Frame 0 (alpha=0.5) and Sub-Frame 1 (alpha=1.0) on the render thread:
//   * Free Follow Camera mode (Vanilla sub_823039D0, Modern sub_83A8FCB8, and
//     OT/UT Free Camera): steps visual yaw/pitch using real-time sub-frame
//     mouse/right-stick input with 120 Hz converted damping k_120 = 1 - sqrt(1 - k_60),
//     interpolates target pivot translation across [Cam[n-1], Cam[n]], and rebuilds
//     View / Proj / ViewProj + auxiliary projection matrices into Camera,
//     CameraManager, and the active render-thread double buffer slot.
//   * Action Camera (sub_82312F80) and Cutscenes (InCutscene()): interpolates
//     Cam[n-1] and Cam[n] with camera-cut threshold protection so shot transitions
//     snap cleanly without ghosting.
// - Exposes ReprojectScreenPoint() so aspect.cpp keeps 3D-projected 2D HUD items
//   (e.g. bow aiming reticle) locked to the live 120 Hz ViewProj matrix.

#pragma once

#include <cstdint>

namespace ng2::high_fps::camera {

// Called around sub_822FA8C8 on the 60 Hz logic thread.
void OnFollowCameraTick(uint32_t camPtr, bool modern, bool suppressedOrLocked);
void OnActionCameraTick(uint32_t camPtr, bool otUtFreeCam);
void OnCameraUpdateEnd(uint32_t camPtr);

// Called at sub_822F0CA8 right before the main loop hands the finished 60 Hz
// logic frame to the render thread (single-threaded boundary, race-free).
void OnLogicHandoff();

// Called on the render thread right before each sub-frame draw.
// dtSec is real elapsed time since the previous sub-frame (1/120 in Fixed 120 mode).
void ApplySubFrameCamera(bool isFirstPass, float alpha, float dtSec);

// Restores the 60 Hz Camera / CameraManager state after both sub-frames finish
// (before returning control to 60 Hz logic) so logic simulation reads clean
// Cam[n] values without drift.
void RestoreLogicCamera();

// Re-projects a 2D point in 1280x720 virtual screen coordinates from the 60 Hz
// logic frame's ViewProj matrix to the active 120 Hz sub-frame's ViewProj matrix.
// Returns true if shifted (outputs outX, outY in 1280x720 space).
bool ReprojectScreenPoint(float x60, float y60, float* outX120, float* outY120);

}  // namespace ng2::high_fps::camera
