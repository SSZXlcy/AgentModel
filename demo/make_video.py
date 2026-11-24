"""5-minute demo video from the report HTML deck (screenshots) and the architecture clips, with burned-in
Chinese subtitles. Run from this directory:  python make_video.py
Outputs: out/智训引擎演示.mp4 (subtitles burned in), out/智训引擎演示.srt, out/storyboard.md"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FFMPEG = str(HERE.parents[1] / "tools" / "ffmpeg.exe")
CLIPS = HERE.parents[1] / "report" / "架构动画"
FONT = "C:/Windows/Fonts/msyhbd.ttc"
BG = (13, 13, 13)
W, H, FPS = 1920, 1080, 30
OUT = HERE / "out"

# (visual, seconds, subtitle lines). visual: ("slide", index) or ("clip", file, title)
STORY = [
    (("slide", 0), 14, ["工业数据一直在变，模型该不该一直训练？",
                        "我们做了「智训引擎」：一套受控自进化的工业模型训练系统。",
                        "一句话：先看清数据，再决定打法；改了先验证，用了要算钱。"]),
    (("slide", 1), 16, ["赛题的分数，看的是每一份成本换来多少提升。",
                        "成本有三样：训练算力、标注答案、大模型 Token。",
                        "还有三道门槛：不崩溃、记录完整、不低于静态基线。"]),
    (("slide", 2), 28, ["我们先试了最简单的做法：每来一批数据，就接着训练。",
                        "气体任务上，它提升了 0.31；",
                        "轴承任务上，它反而让误差多了 17 分钟；",
                        "发动机任务上，花了 50 万行的训练量，提升还在噪声以内。",
                        "同一招，三种结局——因为三类数据的脾气完全不同。",
                        "气体：传感器一直在漂，需要我们「追」。",
                        "轴承：寿命要等坏了才知道，需要我们「等」。",
                        "发动机：新数据早被训练集覆盖，需要我们「省」。"]),
    (("slide", 3), 24, ["每一条设计，都来自一次实测。",
                        "在刚训练过的数据上评估，候选会被高估 0.28；",
                        "不按设备分组，噪声会被低估 21 倍。",
                        "我们也踩过坑：早期报过「轴承提升 69.5 分钟」，",
                        "后来发现模拟时提前拿到了答案，我们主动撤回了。"]),
    (("slide", 4), 16, ["系统分三层：策略层决定怎么改，训练层每批数据都运行，",
                        "执行层只调用官方训练器。",
                        "工业模型始终由官方程序产出，默认插槽下与官方逐字节相同。"]),
    (("slide", 5), 12, ["第一件事，看清数据。",
                        "第一批数据到来时，系统只看输入、不看答案，",
                        "自动认出谁是设备、谁是时间、答案是不是倒计时。"]),
    (("clip", "①开局.mp4", "① 开局：读结构、选节奏、做开局实验"), 12,
     ["再算出答案来得及不及时、输入新不新，",
      "并在训练集上按设备留出做开局实验，只让验证通过的修改生效。"]),
    (("clip", "②每批数据.mp4", "② 每批数据：预测 → 买答案 → 训练候选 → 过门槛"), 12,
     ["之后每来一批数据：先预测，再按节奏买少量答案，",
      "候选必须在没见过的设备上赢过噪声，才能上线试用。"]),
    (("slide", 6), 16, ["第二件事，选对打法。",
                        "气体积极跟随；轴承等设备跑完再学；发动机少动、省钱。",
                        "这三种节奏，都是系统根据数据特征自己选的。"]),
    (("clip", "③监控与策略.mp4", "③ 监控与策略：五项健康检查 → 延续 / 复查 / 更新"), 12,
     ["监督器每批做五项健康检查：",
      "没问题就延续，有疑点就复查，确实失守才更新策略。"]),
    (("slide", 7), 16, ["第三件事，改得稳。",
                        "所有修改都来自一本「修改说明书」，里面有 32 个算子。",
                        "经验写成「什么情况 → 用哪个算子」，换个任务也能直接用。"]),
    (("clip", "④闭环学习.mp4", "④ 闭环学习：观察期 → 确认或撤销 → 写回经验库"), 12,
     ["修改生效后还要过观察期，变差就撤销，",
      "结果再写回经验库，调整这条经验的可信度。"]),
    (("slide", 8), 18, ["结果如何？先看 7 条公开数据流。",
                        "没有一条低于静态基线超过波动范围，运行 0 次崩溃。",
                        "轴承在现实的答案时机下降低误差 52.8 分钟，无脑更新反而多了 17 分钟。"]),
    (("slide", 9), 18, ["系统冻结后，我们做了一次性核验，不据结果调参。",
                        "三根隐藏轴承的误差从 581 分钟降到 224 分钟，降低 61%；",
                        "同样的数据上，无脑更新和只有训练层几乎没动。"]),
    (("slide", 10), 14, ["同等或更好的效果，花得更少。",
                         "在发动机上，我们是唯一没有变差的做法，标注还少了八成。"]),
    (("slide", 11), 16, ["大模型只在规则不够用的时候出场。",
                         "27B 的 Token 降到原来的 2%，分数反而从 0.173 升到 0.249。"]),
    (("slide", 12), 14, ["任务配置只写数据位置；把字段全部改名，决策不变；",
                         "先在两个任务上积累经验再跑第三个，决策也一致。"]),
    (("slide", 13), 14, ["我们也清楚短板：在强漂移的气体任务上，还略低于无脑更新。",
                         "原因已经找到：安全网按单批回退，偏谨慎。"]),
    (("slide", 14), 16, ["我们做的不只是一个更会训练的模型，",
                         "而是一个知道什么时候该训练、怎么训练、训练多少才值得的系统。",
                         "谢谢观看。"]),
]
BAND = 108                      # bottom band kept free for subtitles (slides are scaled to 90%)


def run(*args: str) -> None:
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args], check=True, cwd=HERE)


def slide_frame(index: int, path: Path) -> None:
    shot = Image.open(HERE / "shots" / f"s{index}.png").convert("RGB")
    scale = (H - BAND) / shot.height
    shot = shot.resize((round(shot.width * scale), H - BAND), Image.LANCZOS)
    frame = Image.new("RGB", (W, H), BG)
    frame.paste(shot, ((W - shot.width) // 2, 0))
    frame.save(path)


def clip_background(title: str, path: Path) -> None:
    frame = Image.new("RGB", (W, H), (16, 21, 30))
    draw = ImageDraw.Draw(frame)
    font = ImageFont.truetype(FONT, 40)
    box = draw.textbbox((0, 0), title, font=font)
    draw.text(((W - (box[2] - box[0])) // 2, 34), title, font=font, fill=(235, 238, 242))
    draw.rectangle([(W - 1200) // 2 - 2, 104, (W + 1200) // 2 + 1, 104 + 840 + 1], outline=(60, 70, 86), width=2)
    frame.save(path)


def stamp(seconds: float) -> tuple[str, str]:
    cs = round(seconds * 100)
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}", f"{h:02d}:{m:02d}:{s:02d},{cs * 10:03d}"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    seg_dir = HERE / "segments"
    shutil.rmtree(seg_dir, ignore_errors=True)
    seg_dir.mkdir()
    cues, t, board = [], 0.0, ["| # | 时间 | 画面 | 字幕 |", "|---|---|---|---|"]
    for n, (visual, seconds, lines) in enumerate(STORY):
        segment = seg_dir / f"{n:02d}.mp4"
        fade = f"fade=in:st=0:d=0.35:color=0x0d0d0d,fade=out:st={seconds - 0.3}:d=0.3:color=0x0d0d0d"
        if visual[0] == "slide":
            still = seg_dir / f"{n:02d}.png"
            slide_frame(visual[1], still)
            run("-loop", "1", "-framerate", str(FPS), "-i", str(still.relative_to(HERE)), "-t", str(seconds),
                "-vf", f"{fade},format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-r", str(FPS), str(segment.relative_to(HERE)))
            label = f"展示页第 {visual[1] + 1} 页"
        else:
            bg = seg_dir / f"{n:02d}_bg.png"
            clip_background(visual[2], bg)
            clip = CLIPS / visual[1]
            run("-loop", "1", "-framerate", str(FPS), "-i", str(bg.relative_to(HERE)),
                "-stream_loop", "-1", "-i", str(clip),
                "-filter_complex", f"[1:v]scale=1200:840:flags=lanczos[c];[0:v][c]overlay={(W - 1200) // 2}:105,{fade},format=yuv420p",
                "-t", str(seconds), "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-r", str(FPS),
                str(segment.relative_to(HERE)))
            label = f"架构动画 {visual[1].removesuffix('.mp4')}"
        weights = [max(len(line), 8) for line in lines]
        start = t + 0.25
        span = seconds - 0.5
        for line, w in zip(lines, weights):
            d = span * w / sum(weights)
            cues.append((start, start + d - 0.08, line))
            start += d
        board.append(f"| {n + 1} | {stamp(t)[1][3:8]}–{stamp(t + seconds)[1][3:8]} | {label} | {' / '.join(lines)} |")
        t += seconds
        print(f"segment {n + 1}/{len(STORY)} done", flush=True)

    (seg_dir / "list.txt").write_text("".join(f"file '{n:02d}.mp4'\n" for n in range(len(STORY))), encoding="utf-8")
    run("-f", "concat", "-safe", "0", "-i", "segments/list.txt", "-c", "copy", "segments/joined.mp4")

    srt = "".join(f"{i}\n{stamp(a)[1]} --> {stamp(b)[1]}\n{text}\n\n" for i, (a, b, text) in enumerate(cues, 1))
    (OUT / "智训引擎演示.srt").write_text(srt, encoding="utf-8-sig")
    ass = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 0", "",
           "[V4+ Styles]",
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
           "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
           "MarginR, MarginV, Encoding",
           "Style: Sub,Microsoft YaHei,46,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,1,0,1,3,1,2,80,80,32,134",
           "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    ass += [f"Dialogue: 0,{stamp(a)[0]},{stamp(b)[0]},Sub,,0,0,0,,{text}" for a, b, text in cues]
    (seg_dir / "subs.ass").write_text("\n".join(ass) + "\n", encoding="utf-8")
    fonts = seg_dir / "fonts"
    fonts.mkdir(exist_ok=True)
    for name in ("msyh.ttc", "msyhbd.ttc"):
        shutil.copyfile(f"C:/Windows/Fonts/{name}", fonts / name)
    target = OUT / "智训引擎演示.mp4"
    run("-i", "segments/joined.mp4", "-vf", "subtitles=segments/subs.ass:fontsdir=segments/fonts",
        "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "out/tmp.mp4")
    (OUT / "tmp.mp4").replace(target)
    (OUT / "storyboard.md").write_text("# 演示视频分镜与字幕\n\n" + "\n".join(board) + "\n", encoding="utf-8")
    print(f"total {t:.0f}s, {len(cues)} subtitle lines -> {target}")


if __name__ == "__main__":
    main()
