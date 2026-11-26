// ng2 - Ninja Gaiden II static recompilation
//
// 120 Hz host input sampling, button edge latching, and sub-frame camera
// look delta distribution for decoupled native 120 FPS rendering.

#pragma once

#include <cstdint>

namespace ng2::high_fps::input {

// Latest sub-frame right-stick / mouse sample consumed by 120 Hz camera stepping.
struct CameraInputSample {
  int32_t rx = 0;            // [-32768, 32767] raw right-stick or half-frame mouse delta
  int32_t ry = 0;
  bool isMouse = false;      // true when MnkInputDriver mouse delta supplied this sample
  int64_t timestampNs = 0;   // Steady clock sample time for NG2_LATENCY_LOG
  uint64_t seq = 0;          // Monotonically incremented on each 120 Hz poll
};

// Last consolidated sample delivered to the 60 Hz logic tick (sub_822F2E38).
struct LogicInputSample {
  int32_t rx = 0;
  int32_t ry = 0;
  int32_t unclampedRx = 0;   // Un-clamped mouse accumulator sum for camera calibration
  int32_t unclampedRy = 0;
  bool isMouse = false;
  uint64_t logicFrameId = 0;
};

// Polls host controllers & MNK at 120 Hz on render sub-frame start:
//  - OR-latches rising button/trigger edges into the pending 60 Hz logic frame
//  - Accumulates mouse deltas (consumed on read by MnkInputDriver) for 60 Hz logic
//  - Captures the fresh half-frame look delta/stick deflection for 120 Hz camera
void PollSubFrame(bool isSubFrame0);

// Returns the latest 120 Hz sub-frame look sample (lock-free copy).
CameraInputSample LatestCameraInput();

// Returns the consolidated look input delivered to the most recent 60 Hz logic tick.
LogicInputSample LatestLogicInput();

}  // namespace ng2::high_fps::input
