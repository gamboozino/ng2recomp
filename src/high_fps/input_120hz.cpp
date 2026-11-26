// ng2 - Ninja Gaiden II static recompilation
//
// 120 Hz host input polling & edge latching:
// - Polls InputSystem::GetState at 120 Hz (on both render Sub-Frame 0 and
//   Sub-Frame 1) and right before 60 Hz sub_822F2E38 reads XamInputGetState.
// - Rising button edges between 60 Hz logic ticks are OR-latched so quick taps
//   landing inside a single 8.33 ms sub-frame are never dropped by 60 Hz logic.
// - MnkInputDriver resets mouse_dx_/mouse_dy_ on every GetState() call, so
//   mouse deltas are split cleanly: each half-frame gets its own instantaneous
//   delta for 120 Hz camera stepping, while the sum across sub-frames is fed to
//   60 Hz logic so gameplay camera convergence matches 1:1.

#include "high_fps/input_120hz.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>

#include <rex/hook.h>
#include <rex/input/input.h>
#include <rex/input/input_system.h>
#include <rex/ppc/context.h>
#include <rex/runtime.h>
#include <rex/types.h>

#include "controls.h"
#include "gpu/guest.h"
#include "gpu/video.h"
#include "high_fps/high_fps.h"

namespace ng2::high_fps::input {
namespace {

constexpr uint32_t kMaxUsers = 4;
constexpr uint8_t kTriggerDigitalThreshold = 30;
constexpr uint32_t kErrorDeviceNotConnected = 1167;

struct UserSlotState {
  uint32_t lastResult = kErrorDeviceNotConnected;
  rex::input::X_INPUT_STATE lastRaw{};
  uint16_t prevRawButtons = 0;
  uint16_t latchedPress = 0;
  uint8_t prevRawLt = 0;
  uint8_t prevRawRt = 0;
  uint8_t latchedLtEdge = 0;
  uint8_t latchedRtEdge = 0;
  int32_t mouseAccumRx = 0;
  int32_t mouseAccumRy = 0;
  bool mouseAccumUsed = false;
  int32_t camMousePendingRx = 0;
  int32_t camMousePendingRy = 0;
};

std::mutex g_mutex;
UserSlotState g_slots[kMaxUsers];
CameraInputSample g_latestCamera{};
LogicInputSample g_latestLogic{};
uint64_t g_pollSeq = 0;

int64_t SteadyNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

rex::input::InputSystem* GetInputSystem() {
  auto* runtime = rex::Runtime::instance();
  return runtime ? static_cast<rex::input::InputSystem*>(runtime->input_system()) : nullptr;
}

// Polls one user slot under g_mutex. Returns whether a fresh state was read.
bool PollUserLocked(rex::input::InputSystem* input, uint32_t userIndex, bool isMouseActive,
                    int32_t* outSubRx = nullptr, int32_t* outSubRy = nullptr) {
  if (!input || userIndex >= kMaxUsers) {
    return false;
  }
  rex::input::X_INPUT_STATE raw{};
  rex::X_RESULT res = input->GetState(userIndex, &raw);
  UserSlotState& slot = g_slots[userIndex];
  slot.lastResult = res;

  if (res != 0) {
    slot.lastRaw = {};
    slot.prevRawButtons = 0;
    slot.prevRawLt = 0;
    slot.prevRawRt = 0;
    if (outSubRx) *outSubRx = 0;
    if (outSubRy) *outSubRy = 0;
    return false;
  }

  slot.lastRaw = raw;
  const uint16_t buttons = uint16_t(raw.gamepad.buttons);
  const uint8_t lt = raw.gamepad.left_trigger;
  const uint8_t rt = raw.gamepad.right_trigger;
  const int32_t rx = int32_t(int16_t(raw.gamepad.thumb_rx));
  const int32_t ry = int32_t(int16_t(raw.gamepad.thumb_ry));

  // OR-latch any rising edge across sub-frames so 60 Hz logic never drops taps.
  slot.latchedPress |= uint16_t(buttons & ~slot.prevRawButtons);
  slot.prevRawButtons = buttons;

  if (lt >= kTriggerDigitalThreshold && slot.prevRawLt < kTriggerDigitalThreshold) {
    slot.latchedLtEdge = std::max(slot.latchedLtEdge, lt);
  }
  if (rt >= kTriggerDigitalThreshold && slot.prevRawRt < kTriggerDigitalThreshold) {
    slot.latchedRtEdge = std::max(slot.latchedRtEdge, rt);
  }
  slot.prevRawLt = lt;
  slot.prevRawRt = rt;

  if (isMouseActive && userIndex == 0) {
    slot.mouseAccumRx += rx;
    slot.mouseAccumRy += ry;
    slot.mouseAccumUsed = true;
    slot.camMousePendingRx += rx;
    slot.camMousePendingRy += ry;
  } else if (userIndex == 0) {
    slot.camMousePendingRx = 0;
    slot.camMousePendingRy = 0;
  }

  if (outSubRx) *outSubRx = rx;
  if (outSubRy) *outSubRy = ry;
  return true;
}

// Builds the 60 Hz guest X_INPUT_STATE for userIndex, draining latched edges and
// accumulated mouse deltas accumulated since the previous logic tick.
uint32_t ConsumeUserLocked(rex::input::InputSystem* input, uint32_t userIndex, bool isMouseActive,
                           rex::input::X_INPUT_STATE* outState) {
  PollUserLocked(input, userIndex, isMouseActive);
  UserSlotState& slot = g_slots[userIndex];
  if (slot.lastResult != 0) {
    return slot.lastResult;
  }

  rex::input::X_INPUT_STATE merged = slot.lastRaw;
  const uint16_t buttons = uint16_t(slot.lastRaw.gamepad.buttons) | slot.latchedPress;
  const uint8_t lt = std::max(slot.lastRaw.gamepad.left_trigger, slot.latchedLtEdge);
  const uint8_t rt = std::max(slot.lastRaw.gamepad.right_trigger, slot.latchedRtEdge);

  merged.gamepad.buttons = buttons;
  merged.gamepad.left_trigger = lt;
  merged.gamepad.right_trigger = rt;

  int32_t finalRx = int32_t(int16_t(slot.lastRaw.gamepad.thumb_rx));
  int32_t finalRy = int32_t(int16_t(slot.lastRaw.gamepad.thumb_ry));
  int32_t unclampedRx = finalRx;
  int32_t unclampedRy = finalRy;
  bool deliveredMouse = false;
  if (userIndex == 0 && slot.mouseAccumUsed) {
    unclampedRx = slot.mouseAccumRx;
    unclampedRy = slot.mouseAccumRy;
    finalRx = std::clamp(slot.mouseAccumRx, -32768, 32767);
    finalRy = std::clamp(slot.mouseAccumRy, -32768, 32767);
    merged.gamepad.thumb_rx = int16_t(finalRx);
    merged.gamepad.thumb_ry = int16_t(finalRy);
    deliveredMouse = isMouseActive;
  }

  if (userIndex == 0) {
    g_latestLogic.rx = finalRx;
    g_latestLogic.ry = finalRy;
    g_latestLogic.unclampedRx = unclampedRx;
    g_latestLogic.unclampedRy = unclampedRy;
    g_latestLogic.isMouse = deliveredMouse;
    g_latestLogic.logicFrameId = ng2::high_fps::LogicFrameId();
  }

  // Clear latches after 60 Hz logic consumes them.
  slot.latchedPress = 0;
  slot.latchedLtEdge = 0;
  slot.latchedRtEdge = 0;
  slot.mouseAccumRx = 0;
  slot.mouseAccumRy = 0;
  slot.mouseAccumUsed = false;

  if (outState) {
    *outState = merged;
  }
  return 0;
}

}  // namespace

void PollSubFrame(bool /*isSubFrame0*/) {
  if (!ng2::high_fps::IsActive()) {
    return;
  }
  auto* input = GetInputSystem();
  if (!input) {
    return;
  }

  const bool isMouse = ng2::controls::ActiveDevice() == ng2::controls::Device::kKeyboard;
  const int64_t nowNs = SteadyNs();

  std::lock_guard<std::mutex> lock(g_mutex);
  int32_t camRx = 0, camRy = 0;
  bool gotCamSample = false;

  for (uint32_t u = 0; u < kMaxUsers; u++) {
    int32_t rx = 0, ry = 0;
    if (PollUserLocked(input, u, isMouse, &rx, &ry)) {
      if (u == 0 || (!gotCamSample && (rx != 0 || ry != 0))) {
        camRx = rx;
        camRy = ry;
        gotCamSample = true;
      }
    }
  }

  if (isMouse) {
    camRx = g_slots[0].camMousePendingRx;
    camRy = g_slots[0].camMousePendingRy;
    g_slots[0].camMousePendingRx = 0;
    g_slots[0].camMousePendingRy = 0;
  }

  g_latestCamera.rx = camRx;
  g_latestCamera.ry = camRy;
  g_latestCamera.isMouse = isMouse;
  g_latestCamera.timestampNs = nowNs;
  g_latestCamera.seq = ++g_pollSeq;

  // Refresh latency tracking per 120 Hz sub-frame so NG2_LATENCY_LOG=1 measures
  // true sub-frame sample -> present -> screen latency for both Sub-Frame 0 & 1.
  ng2::video::NoteInputRead();
  ng2::video::NoteFrameTaken();
}

CameraInputSample LatestCameraInput() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_latestCamera;
}

LogicInputSample LatestLogicInput() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_latestLogic;
}

}  // namespace ng2::high_fps::input

// Hooks the guest XamInputGetState thunk (called per pad inside sub_837F0718
// from 60 Hz sub_822F2E38) to deliver OR-latched buttons and accumulated mouse
// deltas when 120 Hz input sampling is active.
REX_EXTERN(__imp__sub_83991FA8);
REX_HOOK_RAW(sub_83991FA8) {
  if (!ng2::high_fps::IsActive()) {
    __imp__sub_83991FA8(ctx, base);
    return;
  }

  const uint32_t userIndex = ctx.r3.u32;
  const uint32_t guestStatePtr = ctx.r4.u32;
  if (userIndex >= ng2::high_fps::input::kMaxUsers || !guestStatePtr) {
    __imp__sub_83991FA8(ctx, base);
    return;
  }

  auto* input = ng2::high_fps::input::GetInputSystem();
  if (!input) {
    __imp__sub_83991FA8(ctx, base);
    return;
  }

  const bool isMouse = ng2::controls::ActiveDevice() == ng2::controls::Device::kKeyboard;
  rex::input::X_INPUT_STATE outState{};
  uint32_t res = 0;
  {
    std::lock_guard<std::mutex> lock(ng2::high_fps::input::g_mutex);
    res = ng2::high_fps::input::ConsumeUserLocked(input, userIndex, isMouse, &outState);
  }
  if (res == 0) {
    std::memcpy(ng2::guest::Virtual(guestStatePtr), &outState, sizeof(outState));
  }
  ctx.r3.s64 = int64_t(res);
}
