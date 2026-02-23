"""ボーカル分離ツール: demucs 4ステム分離.

htdemucs モデルを使って楽曲を vocals/drums/bass/other の4ステムに分離する。
オンデマンド実行。出力WAVをキャッシュに保存してパスを返す。
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from pathlib import Path

import librosa

# キャッシュディレクトリ（loader と同じルート下）
_STEMS_CACHE_DIR = Path(tempfile.gettempdir()) / "vml_audio_lab" / "stems"

STEM_NAMES = ("vocals", "drums", "bass", "other")
_DEFAULT_MODEL = "htdemucs"
_DEFAULT_SR = 44100


def _stem_cache_dir(source_path: str, model_name: str = _DEFAULT_MODEL) -> Path:
    """ソースパスとモデル名からステムキャッシュディレクトリを返す。"""
    key = hashlib.sha256(f"{source_path}|{model_name}".encode()).hexdigest()[:16]
    return _STEMS_CACHE_DIR / key


def _all_stems_cached(cache_dir: Path) -> bool:
    """全ステムがキャッシュ済みかどうか確認する。"""
    return all((cache_dir / f"{stem}.wav").exists() for stem in STEM_NAMES)


def _demucs_separate(wav_path: str, output_dir: str, model_name: str = _DEFAULT_MODEL) -> dict[str, str]:
    """demucs モデルで音声を4ステムに分離する。

    テストでのモック差し替えを想定した内部関数。

    Args:
        wav_path: 分離対象の音声ファイルパス
        output_dir: ステムWAV出力先ディレクトリ
        model_name: demucs モデル名

    Returns:
        stem名 → WAVパスの辞書 (vocals, drums, bass, other)
    """
    import soundfile as sf
    import torch
    import torchaudio.transforms
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[separator] Loading model: {model_name} ...")
    model = get_model(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[separator] Running on {device}")

    # soundfile で音声読み込み (torchcodec 依存の torchaudio.load を回避)
    audio_np, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    # soundfile は [samples, channels] → [channels, samples] に転置
    wav = torch.tensor(audio_np.T)
    # モノラルの場合はステレオに変換
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    # モデルのサンプリングレートにリサンプル
    if sr != model.samplerate:
        resampler = torchaudio.transforms.Resample(sr, model.samplerate)
        wav = resampler(wav)

    # バッチ次元を追加: [channels, samples] → [1, channels, samples]
    wav = wav.unsqueeze(0).to(device)

    print("[separator] Separating stems ...")
    with torch.no_grad():
        sources = apply_model(model, wav, device=device)

    # sources shape: [1, n_stems, channels, samples]
    sources = sources[0]  # [n_stems, channels, samples]

    model_stem_names: list[str] = list(model.sources)
    sr_out = model.samplerate

    result: dict[str, str] = {}
    for i, stem_name in enumerate(model_stem_names):
        if stem_name in STEM_NAMES:
            stem_wav = sources[i].cpu()  # [channels, samples]
            out_path = out_dir / f"{stem_name}.wav"
            # soundfile で保存 (torchaudio.save の torchcodec 依存を回避)
            # soundfile expects [samples, channels]
            sf.write(str(out_path), stem_wav.numpy().T, sr_out, subtype="PCM_16")
            result[stem_name] = str(out_path)
            print(f"[separator] Saved {stem_name} → {out_path}")

    # GPU メモリ解放
    del model, wav, sources
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def separate_stems(
    file_path: str,
    model_name: str = _DEFAULT_MODEL,
) -> dict:
    """音声を4ステム（vocals/drums/bass/other）に分離する。

    htdemucs モデルを使用。初回は数分かかる場合がある。
    ステムはキャッシュに保存され、2回目以降は即時返却。

    Args:
        file_path: 音声ファイルのローカルパス (wav/mp3/flac 等)
        model_name: demucs モデル名（デフォルト: "htdemucs"）

    Returns:
        dict:
            - stems: 各ステムの WAVパス辞書 (vocals, drums, bass, other)
            - model: 使用したモデル名
            - sample_rate: 出力サンプリングレート
            - duration_sec: 楽曲長（秒）
            - processing_time_sec: 処理時間（秒）、キャッシュヒット時は 0.0
            - source_file: 元ファイルのパス
            - cached: キャッシュから返した場合 True

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        RuntimeError: demucs 処理失敗時
    """
    src = Path(file_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"音声ファイルが見つかりません: {src}")

    cache_dir = _stem_cache_dir(str(src), model_name)

    # キャッシュヒット: 音声ロードをスキップして即時返却
    if _all_stems_cached(cache_dir):
        stems = {stem: str(cache_dir / f"{stem}.wav") for stem in STEM_NAMES}
        # キャッシュヒット時の楽曲長は soundfile で高速取得
        try:
            import soundfile as sf
            info = sf.info(str(src))
            duration_sec = float(info.duration)
            sr_out = int(info.samplerate)
        except Exception:
            duration_sec = 0.0
            sr_out = _DEFAULT_SR
        return {
            "stems": stems,
            "model": model_name,
            "sample_rate": sr_out,
            "duration_sec": round(duration_sec, 2),
            "processing_time_sec": 0.0,
            "source_file": str(src),
            "cached": True,
        }

    # 楽曲長を取得 (キャッシュミス時のみ)
    y, sr = librosa.load(str(src), sr=None, mono=True, duration=10.0)
    # torchaudio でファイル全体のフレーム数を取得
    try:
        import torchaudio
        info = torchaudio.info(str(src))
        duration_sec = float(info.num_frames) / float(info.sample_rate)
        sr_out = info.sample_rate
    except Exception:
        duration_sec = float(librosa.get_duration(y=y, sr=sr))
        sr_out = _DEFAULT_SR

    cache_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    stem_paths = _demucs_separate(str(src), str(cache_dir), model_name)
    processing_time = time.time() - start_time

    # 不足ステムを補完
    stems: dict[str, str] = {stem: "" for stem in STEM_NAMES}
    stems.update({k: v for k, v in stem_paths.items() if k in STEM_NAMES})

    return {
        "stems": stems,
        "model": model_name,
        "sample_rate": sr_out,
        "duration_sec": round(duration_sec, 2),
        "processing_time_sec": round(processing_time, 1),
        "source_file": str(src),
        "cached": False,
    }


def extract_vocals(file_path: str) -> dict:
    """ボーカルを含む全ステムを分離して返す（`separate_stems` の薄いラッパー）。

    `separate_stems` を呼び出し、stems を展開したフラットな dict を返す。
    テスト・後方互換のため維持。

    Args:
        file_path: 音声ファイルのローカルパス (wav/mp3/flac 等)

    Returns:
        dict:
            - vocals: ボーカルステムのWAVパス
            - drums: ドラムステムのWAVパス
            - bass: ベースステムのWAVパス
            - other: その他ステムのWAVパス
            - model: 使用したモデル名
            - cached: キャッシュから返した場合 True

    Raises:
        FileNotFoundError: ファイルが存在しない場合
    """
    result = separate_stems(file_path)
    stems = result["stems"]
    return {
        **stems,
        "model": result["model"],
        "cached": result["cached"],
    }
