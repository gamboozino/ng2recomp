// ng2 - Ninja Gaiden II static recompilation
//
// The Cheats page of RECOMP SETTINGS (recomp_settings.cpp).
//
// Infinite Health: damage to a character goes through sub_82412508 (r3 a
// record whose first word is the character), and the damage that lowers its
// most health (the red end of the bar) through sub_82412EA8 and
// sub_824131E0. Each gives none when the byte at 0x83CB5992 is 1 (anyone),
// or 2 and the character is the player (two of the character's own flags,
// and in sub_82412508 the record's +10 being 0). It is set to 2 for those
// calls only, so enemies still take theirs.
//
// Infinite Ninpo: a character's ninpo is a u16 at +0x5942 (200 a charge),
// its most at +0x5940. A cast spends as usual, and the player's ninpo is
// filled back to its most each frame in play (CheatsFrame, from the frame's
// start). (A cost of 0 instead left the HUD
// showing a charge spent: its gauge plays a cast's drain itself and only
// shows the charges again when it sees the ninpo go up.)
//
// Unlock Difficulties: sub_823E1098(clears, d) says difficulty d is open
// (some difficulty at or above d - 1 cleared). The New Game path list
// (sub_8366F4D0) asks it for Mentor and Master Ninja; the answer is yes for
// those two calls. Mission Mode doesn't ask: choosing a mission
// (sub_823E4758) sets how many difficulties it offers (1, 2, 3 or 4) from
// the mission's record in the profile (cleared flags 0x10..0x80 in byte 7
// of [0x84C559E0] + 1016 + 8 * mission), except for missions whose table
// entry (0x83B2E334 + 12 * mission) is 3, which offer one. A hook right
// after the count is set raises it to 4.
//
// Unlock Costumes: the costume list (sub_823E1448) adds a costume per
// difficulty the same check says is cleared; it is yes there too.
//
// Neither writes the unlocks to the profile (a game played on a difficulty
// or in a costume they open is saved as usual): switched off, the menus
// are as before.
//
// Always Cut Limbs: a reaction rule can ask for a chance roll:
// sub_823A95E0(r3 attacker record, r4 kind, f1 scale) reads a percentage
// from the attack's data (sub_823A81C8(record, 12, attack), the u16 at +12)
// and rolls rand() * 100 / 32768 against it (0: never, 100: always). The
// cut reactions are such rolls, high for strong attacks and low (or 0) for
// light ones. For the player's hits (the attacker is the player's record)
// the attack's chance is set to 100 and the roll's answer is yes. Forcing
// the reaction rule instead broke Obliteration. The chances are put back
// when it's turned off.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include <rex/cvar.h>
#include <rex/hook.h>
#include <rex/logging.h>

#include "aspect.h"
#include "gpu/guest.h"

REXCVAR_DEFINE_BOOL(cheat_infinite_health, false, "Cheats", "The player takes no damage");
REXCVAR_DEFINE_BOOL(cheat_infinite_ninpo, false, "Cheats", "Casting ninpo doesn't spend it");
REXCVAR_DEFINE_BOOL(cheat_unlock_difficulties, false, "Cheats",
                    "Every difficulty in New Game and Mission Mode");
REXCVAR_DEFINE_BOOL(cheat_unlock_costumes, false, "Cheats", "Every costume");
REXCVAR_DEFINE_BOOL(cheat_always_cut, false, "Cheats",
                    "The player's attacks always cut a limb where they can");

namespace {

constexpr uint32_t kNoDamage = 0x83CB5992;
constexpr uint8_t kPlayerOnly = 2;

// A damage function with the player's no-damage flag set.
void WithoutPlayerDamage(PPCContext& ctx, uint8_t* base, void (*damage)(PPCContext&, uint8_t*)) {
  uint8_t* flag = ng2::guest::Virtual(kNoDamage);
  uint8_t game = *flag;
  bool ours = REXCVAR_GET(cheat_infinite_health) && game == 0;
  if (ours) {
    *flag = kPlayerOnly;
  }
  damage(ctx, base);
  if (ours && *flag == kPlayerOnly) {
    *flag = game;
  }
}

}  // namespace

REX_EXTERN(__imp__sub_82412508);
REX_HOOK_RAW(sub_82412508) {
  WithoutPlayerDamage(ctx, base, __imp__sub_82412508);
}
REX_EXTERN(__imp__sub_82412EA8);
REX_HOOK_RAW(sub_82412EA8) {
  WithoutPlayerDamage(ctx, base, __imp__sub_82412EA8);
}
REX_EXTERN(__imp__sub_824131E0);
REX_HOOK_RAW(sub_824131E0) {
  WithoutPlayerDamage(ctx, base, __imp__sub_824131E0);
}

// The player's record is the first of the characters' 24-byte records; the
// game reads its character straight from 0x84C3DC60. The game's own test that
// a record's character is the player (sub_82412508): the record's +10 is 0,
// the character's +0x59D3 isn't 0xFF and its +0x59BF is 0.
constexpr uint32_t kPlayerRecord = 0x84C3DC60;
constexpr uint32_t kNinpo = 0x5942, kNinpoMost = 0x5940;

void FillPlayerNinpo();
bool HudInPlay();  // extras.cpp

// The New Game path list is made with the title menu (sub_8366F4D0, at
// 0x853A3470, after the menu's entries: Acolyte and Warrior, then Mentor
// and Master Ninja if open, their count at +8208). It is always made with
// all four, and the count follows the setting (CheatsFrame): the paths the
// profile has opened, or all of them, at once either way.
constexpr uint32_t kPathCount = 0x853A3470 + 8208;
constexpr uint32_t kProfile = 0x84C559E0;
bool g_allPathsAsked = false;  // while the list is made
uint32_t g_allPaths = 0, g_openPaths = 0;

// The unlock check, by caller (its return address).
REX_EXTERN(__imp__sub_823E1098);
REX_HOOK_RAW(sub_823E1098) {
  uint32_t from = uint32_t(ctx.lr);
  bool paths = from == 0x8366F874 || from == 0x8366F8A8;
  bool costumes = from == 0x823E1714 || from == 0x823E174C || from == 0x823E1784 ||
                  from == 0x823E17BC;
  if ((paths && g_allPathsAsked) ||
      (costumes && REXCVAR_GET(cheat_unlock_costumes))) {
    ctx.r3.u64 = 1;
    return;
  }
  __imp__sub_823E1098(ctx, base);
}

// Around the title menu's builder (recomp_settings.cpp's hook): all four
// paths asked for, then the ones open without the cheat counted.
void CheatsTitleMenu(PPCContext& ctx, uint8_t* base, void (*build)(PPCContext&, uint8_t*)) {
  g_allPathsAsked = true;
  build(ctx, base);
  g_allPathsAsked = false;
  g_allPaths = ng2::guest::Load32(kPathCount);
  // The paths open without the cheat (Master Ninja only after Mentor, as
  // the list has them).
  uint32_t profile = ng2::guest::Load32(kProfile);
  uint32_t open = 2;
  if (profile && g_allPaths == 4) {
    PPCRegister r3 = ctx.r3, r4 = ctx.r4;
    for (uint32_t difficulty : {2u, 3u}) {
      ctx.r3.u64 = profile + 720, ctx.r4.u64 = difficulty;
      __imp__sub_823E1098(ctx, base);
      if (ctx.r3.u32 == 0) {
        break;
      }
      open++;
    }
    ctx.r3 = r3, ctx.r4 = r4;
  }
  g_openPaths = std::min(open, g_allPaths);
}

// Mission Mode: r11 the difficulties offered, r31 the mission screen
// (+4: the mission). ng2_codegen.toml places it at 0x823E4A80.
void Ng2MissionDifficulties(PPCRegister& r11, PPCRegister& r31) {
  constexpr uint32_t kMissionTable = 0x83B2E334;  // 12 bytes a mission
  constexpr uint32_t kFixedDifficulty = 3;
  constexpr uint32_t kAll = 4;
  if (!REXCVAR_GET(cheat_unlock_difficulties)) {
    return;
  }
  uint32_t mission = ng2::guest::Load32(r31.u32 + 4);
  if (mission < 100 && ng2::guest::Load32(kMissionTable + 12 * mission) != kFixedDifficulty &&
      r11.u32 < kAll) {
    // The screen keeps the count at +48 (the choice, its cursor's wrap);
    // r11 places the cursor once.
    r11.u64 = kAll;
    uint8_t* count = ng2::guest::Virtual(r31.u32 + 48);
    count[0] = 0, count[1] = 0, count[2] = 0, count[3] = uint8_t(kAll);
  }
}

// Each frame (perf_overlay.cpp's frame start): Infinite Ninpo keeps the
// gauge full.
void CheatsFrame() {
  // Unlock Difficulties: the New Game paths shown (on the title menu: in
  // play the list is gone, and it is made again with the next title menu).
  // Stopped once play starts (the HUD drawn, or its functions run with the HUD
  // hidden).
  if (ng2::aspect::HudShown() || HudInPlay()) {
    g_allPaths = 0;
  }
  if (g_allPaths >= 2 && g_allPaths <= 4) {
    uint32_t count = REXCVAR_GET(cheat_unlock_difficulties) ? g_allPaths : g_openPaths;
    if (ng2::guest::Load32(kPathCount) != count) {
      uint32_t value = rex::byte_swap(count);
      std::memcpy(ng2::guest::Virtual(kPathCount), &value, 4);
    }
  }
  // In play only: before it the record holds something else.
  if (REXCVAR_GET(cheat_infinite_ninpo) && (ng2::aspect::HudShown() || HudInPlay())) {
    FillPlayerNinpo();
  }
}

// The player's ninpo to its most (a cast's spend, a frame in play).
void FillPlayerNinpo() {
  uint32_t player = ng2::guest::Load32(kPlayerRecord);
  if (player < 0x40000000u || player >= 0xA0000000u ||
      ng2::guest::Virtual(kPlayerRecord)[10] != 0 ||
      ng2::guest::Virtual(player)[0x59D3] == 0xFF || ng2::guest::Virtual(player)[0x59BF] != 0) {
    return;
  }
  uint16_t most = ng2::guest::Load16(player + kNinpoMost);
  if (most && most <= 0x2000 && ng2::guest::Load16(player + kNinpo) < most) {
    uint16_t value = rex::byte_swap(most);
    std::memcpy(ng2::guest::Virtual(player + kNinpo), &value, 2);
  }
}

// Always Cut Limbs (see the top).
namespace {

constexpr uint32_t kRecords = 0x84C3DC60;     // 24 bytes a record; the player's first
constexpr uint32_t kTypeTables = 0x83AA56C8;  // by a record's +10; +240: the attack's offset
constexpr uint32_t kChance = 12;              // in an attack's data (u16, percent)

// An attack's data set to 100, with what it held: put back when the setting
// is turned off, where the data still is that attack at 100 (a room's data
// can be gone and its memory reused).
struct CutPatch {
  uint32_t data;
  uint16_t attack, chance;
};
std::vector<CutPatch> g_cutPatches;

void SetChance(uint32_t data, uint16_t chance) {
  uint16_t value = rex::byte_swap(chance);
  std::memcpy(ng2::guest::Virtual(data + kChance), &value, 2);
}

void UndoCutPatches() {
  for (const CutPatch& p : g_cutPatches) {
    if (ng2::guest::Load16(p.data + 4) == p.attack && ng2::guest::Load16(p.data + kChance) == 100) {
      SetChance(p.data, p.chance);
    }
  }
  g_cutPatches.clear();
}

}  // namespace

REX_EXTERN(sub_823A81C8);  // (record, table, id, 0, 0, 0): a data record

// The attack a roll is for, as sub_823A95E0 finds it (the u16 at the
// attacker's character + the type table's +240 offset), and its data.
uint32_t AttackData(PPCContext& ctx, uint8_t* base, uint32_t record) {
  uint8_t type = ng2::guest::Virtual(record)[10];
  uint16_t attack = type == 0xFF ? 0
                                 : ng2::guest::Load16(ng2::guest::Load32(record) +
                                                      ng2::guest::Load32(kTypeTables + 240 + 4 * type));
  PPCRegister r3 = ctx.r3, r4 = ctx.r4, r5 = ctx.r5, r6 = ctx.r6, r7 = ctx.r7, r8 = ctx.r8;
  ctx.r4.u64 = 12, ctx.r5.u64 = attack, ctx.r6.u64 = 0, ctx.r7.u64 = 0, ctx.r8.u64 = 0;
  sub_823A81C8(ctx, base);
  uint32_t data = ctx.r3.u32;
  ctx.r3 = r3, ctx.r4 = r4, ctx.r5 = r5, ctx.r6 = r6, ctx.r7 = r7, ctx.r8 = r8;
  return data;
}

REX_EXTERN(__imp__sub_823A95E0);
REX_HOOK_RAW(sub_823A95E0) {
  bool on = REXCVAR_GET(cheat_always_cut);
  bool player = on && ctx.r3.u32 == kRecords;
  if (!on && !g_cutPatches.empty()) {
    UndoCutPatches();
  }
  if (player) {
    uint32_t data = AttackData(ctx, base, ctx.r3.u32);
    if (data && ng2::guest::Load16(data + kChance) < 100) {
      g_cutPatches.push_back({data, ng2::guest::Load16(data + 4), ng2::guest::Load16(data + kChance)});
      SetChance(data, 100);
    }
  }
  __imp__sub_823A95E0(ctx, base);
  // Every chance roll of the player's hits succeeds, not only the cut ones.
  if (player) {
    ctx.r3.u64 = 1;
  }
}
