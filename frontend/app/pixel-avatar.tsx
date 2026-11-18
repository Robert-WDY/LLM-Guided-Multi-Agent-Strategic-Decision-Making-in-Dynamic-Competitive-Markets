/**
 * Original 16×16 chibi pixel characters, one per agent seat.
 * Each sprite is composed from a shared face/body template plus a
 * per-character hairstyle and accessory; colours come from the agent palette.
 */
import { memo } from "react";

type Palette = Record<string, string>;

// Legend: . transparent · O outline · H hair · h hair shade · S skin · s skin shade
// E eye · W eye highlight · B blush · M mouth · C clothes · c clothes shade · A accent · X accessory
const FACE = [
  "................",
  "................",
  "................",
  "................",
  "................",
  "...OSSSSSSSSO...",
  "..OSSSSSSSSSSO..",
  "..OSEESSSSEESO..",
  "..OSEWSSSSEWSO..",
  "..OSBSSMMSSBSO..",
  "...OSSSSSSSSO...",
  "....OOCCCCOO....",
  "...OCCCAACCCO...",
  "..OCCCCAACCCCO..",
  "..OSCCCCCCCCSO..",
  "...OOcc..ccOO...",
];

type Variant = { name: string; hair: string[]; extra?: Array<[number, number, string]> };

// Each hair layer is drawn over the face template (rows 0..11, "." keeps the face pixel).
const VARIANTS: Variant[] = [
  {
    name: "双马尾",
    hair: [
      "................",
      ".....OOOOOO.....",
      "...OOHHHHHHOO...",
      "..OHHHHHHHHHHO..",
      ".OHHhHHHHHHhHHO.",
      "OHHO.HHhhHH.OHHO",
      "OHHO........OHHO",
      "OHhO........OhHO",
      "OHhO........OhHO",
      ".OHO........OHO.",
      ".OHO........OHO.",
      "..O..........O..",
    ],
    extra: [[1, 4, "X"], [1, 11, "X"]],
  },
  {
    name: "刺猬头",
    hair: [
      "...O..O..O..O...",
      "..OHOOHOOHOOHO..",
      "..OHHHHHHHHHHO..",
      ".OHHHHHHHHHHHHO.",
      ".OHhHHHhHHHhHHO.",
      "..OHH.HHHH.HHO..",
      "..OH........HO..",
      "................",
      "................",
      "................",
      "................",
      "................",
    ],
  },
  {
    name: "猫耳",
    hair: [
      "..OO........OO..",
      "..OXO......OXO..",
      "..OHHOOOOOOHHO..",
      "..OHHHHHHHHHHO..",
      ".OHHhHHHHHHhHHO.",
      ".OHH.HHHHHH.HHO.",
      ".OH..........HO.",
      ".OH..........HO.",
      "................",
      "................",
      "................",
      "................",
    ],
  },
  {
    name: "长直发",
    hair: [
      "................",
      ".....OOOOOO.....",
      "...OOHHHHHHOO...",
      "..OHHHHHHHHHHO..",
      ".OHHHHHHHHHHHHO.",
      ".OHHHH.hh.HHHHO.",
      ".OHH........HHO.",
      ".OHh........hHO.",
      ".OHh........hHO.",
      ".OHH........HHO.",
      ".OHH........HHO.",
      "..OO........OO..",
    ],
  },
  {
    name: "呆毛短发",
    hair: [
      ".......OO.......",
      "......OHHO......",
      "....OOOHOOOO....",
      "..OOHHHHHHHHOO..",
      ".OHHHHhHHhHHHHO.",
      ".OHH.HHHHHH.HHO.",
      "..OH........HO..",
      "................",
      "................",
      "................",
      "................",
      "................",
    ],
  },
  {
    name: "丸子头",
    hair: [
      "......OOOO......",
      ".....OHHHHO.....",
      "......OXXO......",
      "...OOOHHHHOOO...",
      "..OHHHHHHHHHHO..",
      ".OHHhHHHHHHhHHO.",
      ".OH.HHHHHHHH.HO.",
      "..O..........O..",
      "................",
      "................",
      "................",
      "................",
    ],
  },
  {
    name: "护目镜",
    hair: [
      "................",
      "....OOOOOOOO....",
      "...OHHHHHHHHO...",
      "..OHHHHHHHHHHO..",
      ".OXXXXXXXXXXXXO.",
      ".OXAAXXXXXXAAXO.",
      "..OHH......HHO..",
      "................",
      "................",
      "................",
      "................",
      "................",
    ],
  },
  {
    name: "侧马尾",
    hair: [
      "................",
      "....OOOOOO......",
      "..OOHHHHHHOO.OO.",
      "..OHHHHHHHHHOHHO",
      ".OHHhHHHHHHhHOHO",
      ".OHH.HHHHHH.HOHO",
      ".OH..........OHO",
      ".............OhO",
      ".............OHO",
      "..............O.",
      "................",
      "................",
    ],
    extra: [[3, 12, "X"]],
  },
  {
    name: "鸭舌帽",
    hair: [
      "................",
      "....OOOOOOOO....",
      "...OXXXXXXXXO...",
      "..OXXXXAAXXXXO..",
      "..OXXXXXXXXXXOOO",
      "..OOOOOOOOOOOXXO",
      "..OHH......OOOO.",
      "................",
      "................",
      "................",
      "................",
      "................",
    ],
  },
  {
    name: "狐耳",
    hair: [
      ".OO..........OO.",
      ".OXO........OXO.",
      ".OXHO......OHXO.",
      "..OHHOOOOOOHHO..",
      ".OHHHHHHHHHHHHO.",
      ".OHHhHHHHHHhHHO.",
      ".OH.HH....HH.HO.",
      "..O..........O..",
      "................",
      "................",
      "................",
      "................",
    ],
  },
];

const HAIR_TONES = ["#2b2f5c", "#f2c14e", "#e5e7f5", "#7a3b5c", "#3b2a24", "#f08fb0", "#4b5d8f", "#9b6bd6", "#2e3a3f", "#e8823c"];

function shade(hex: string, amount: number) {
  const n = parseInt(hex.slice(1), 16);
  const f = (c: number) => Math.max(0, Math.min(255, Math.round(c + amount * (amount > 0 ? 255 - c : c))));
  const r = f((n >> 16) & 255), g = f((n >> 8) & 255), b = f(n & 255);
  return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, "0")}`;
}

function compose(index: number): string[][] {
  const v = VARIANTS[index % VARIANTS.length];
  const grid = FACE.map((row) => row.split(""));
  v.hair.forEach((row, y) => {
    row.split("").forEach((ch, x) => {
      if (ch !== ".") grid[y][x] = ch;
    });
  });
  v.extra?.forEach(([y, x, ch]) => (grid[y][x] = ch));
  return grid;
}

const SPRITES = VARIANTS.map((_, i) => compose(i));

export const AVATAR_NAMES = VARIANTS.map((v) => v.name);

export const PixelAvatar = memo(function PixelAvatar({
  index,
  color,
  size = 48,
  blink = true,
}: {
  index: number;
  color: string;
  size?: number;
  blink?: boolean;
}) {
  const grid = SPRITES[index % SPRITES.length];
  const hair = HAIR_TONES[index % HAIR_TONES.length];
  const palette: Palette = {
    O: "#151325",
    H: hair,
    h: shade(hair, 0.28),
    S: "#ffe2cc",
    s: "#f4c3a8",
    E: "#1d1b3a",
    W: "#ffffff",
    B: "#ff9db4",
    M: "#b4475e",
    C: color,
    c: shade(color, -0.35),
    A: "#fff6d6",
    X: shade(color, 0.35),
  };
  const rects: React.ReactNode[] = [];
  grid.forEach((row, y) => {
    let x = 0;
    while (x < row.length) {
      const ch = row[x];
      if (ch === ".") {
        x += 1;
        continue;
      }
      let w = 1;
      while (x + w < row.length && row[x + w] === ch) w += 1;
      rects.push(
        <rect
          key={`${x}-${y}`}
          x={x}
          y={y}
          width={w}
          height={1}
          fill={palette[ch]}
          className={blink && (ch === "E" || ch === "W") ? "px-eye" : undefined}
        />,
      );
      x += w;
    }
  });
  return (
    <svg
      className="px-avatar"
      width={size}
      height={size}
      viewBox="0 0 16 16"
      shapeRendering="crispEdges"
      aria-hidden="true"
    >
      {rects}
    </svg>
  );
});
