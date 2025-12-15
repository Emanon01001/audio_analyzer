//! 2つの音声ファイルを読み込み、平均スペクトル（STFT平均）を比較表示するツール。
//!
//! - UI: `eframe/egui` で2ファイル(A/B)の読み込みとパラメータ調整
//! - デコード: `symphonia` で各フォーマット(MP3/FLACなど)をデコードしてf32モノラルへ
//! - リサンプル: 比較しやすいように `rubato` で任意のサンプルレートへ変換（既定48kHz）
//! - 解析: `rustfft` でSTFTを行い、周波数ごとの平均振幅をdB相対で算出
//!
//! 注意:
//! - MP3などはデコード1回ごとのフレーム数が一定でないことがあるため、リサンプル前に
//!   内部バッファへ貯めて固定長(1024)単位で `FftFixedIn` に渡している。
//! - ここでは表示・比較用途のため、振幅は「最大を0dBに正規化した相対dB」。
use anyhow::{anyhow, Context, Result};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use rfd::FileDialog;
use rustfft::{num_complex::Complex32, FftPlanner};
use std::path::{Path, PathBuf};
use symphonia::core::{
    audio::{AudioBuffer, AudioBufferRef, Signal},
    codecs::DecoderOptions,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};
use symphonia::default::{get_codecs, get_probe};

use rubato::{FftFixedIn, Resampler};

#[derive(Clone, Debug, Default)]
struct FileInfo {
    /// 元ファイルのパス（表示用）。
    path: String,
    /// コンテナ名（ここでは拡張子を表示用に使う）。
    container: String,
    /// デコーダが認識したコーデック名（例: mp3, flac）。
    codec: String,
    /// 解析に使ったサンプルレート（`target_sr` に合わせた後）。
    sample_rate: u32,
    /// 解析に使ったチャンネル数（本アプリでは常にモノ=1）。
    channels: usize,
    /// メタ情報から取れた場合の長さ（秒）。取れない場合は0。
    duration_s: f64,
    /// ビットレート（本実装では取得せず `None`）。
    bitrate_kbps: Option<u32>,
    /// タグ（キー/値）。コンテナに含まれていれば表示する。
    tags: Vec<(String, String)>,
    /// ファイルサイズ(MB)（表示用）。
    size_mb: f64,
}

#[derive(Clone, Debug, Default)]
struct Analysis {
    /// 表示に使う基本情報。
    info: FileInfo,
    /// 平均スペクトルの周波数軸（Hz）。
    freqs_hz: Vec<f32>,
    /// 平均スペクトルの相対dB（最大=0dB）。
    mean_db: Vec<f32>,
    /// “周波数成分”としてのピーク上位（Hz, dB）。
    peaks: Vec<(f32 /*Hz*/, f32 /*dB*/)>,

    /// 描画用に軽量化した点列（[Hz, dB]）。0..24kHzに制限している。
    plot_xy: Vec<[f64; 2]>,
}

struct AppState {
    /// 読み込んだファイルAの解析結果。
    a: Option<Analysis>,
    /// 読み込んだファイルBの解析結果。
    b: Option<Analysis>,

    /// 適用中のフォント（表示用）。
    font_status: String,

    /// STFTのFFTサイズ（大きいほど周波数分解能↑、計算量↑）。
    n_fft: usize,
    /// STFTのホップ長（小さいほど時間方向の平均回数↑、計算量↑）。
    hop: usize,
    /// 比較しやすいように変換する目標サンプルレート。
    target_sr: u32,
    /// 解析に使う最大秒数（長いと計算量↑）。
    analyze_seconds: f64,

    /// UIに表示するステータス文字列。
    status: String,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            a: None,
            b: None,
            font_status: "Font: default".to_string(),
            n_fft: 16384,
            hop: 4096,
            target_sr: 48000,
            analyze_seconds: 60.0,
            status: "Ready".to_string(),
        }
    }
}

fn main() -> eframe::Result<()> {
    // ネイティブウィンドウ設定（サイズなど）。
    let mut native = eframe::NativeOptions::default();
    native.viewport = native.viewport.with_inner_size([1100.0, 700.0]);

    // `eframe 0.33` では、App生成クロージャが `Result<Box<dyn App>, _>` を返す。
    eframe::run_native(
        "Audio Spectrum Analyzer (2 files)",
        native,
        Box::new(|cc| {
            let mut app = AppState::default();

            // 以前選択したフォントがあれば、それをデフォルトとして起動時に適用する。
            // ない場合は、実行ファイル横/カレントディレクトリに `default_font.(ttf|otf)` があればそれを使う。
            if let Some(font_path) = load_default_font_path().or_else(find_default_font_candidate) {
                match apply_font_from_file(&cc.egui_ctx, &font_path) {
                    Ok(()) => app.font_status = format!("Font: {} (default)", font_path.display()),
                    Err(e) => app.status = format!("Font load failed: {e:#}"),
                }
            }

            Ok(Box::new(app))
        }),
    )
}

impl eframe::App for AppState {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 上部: 操作パネル（ロード、パラメータ、ステータス）
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Load A").clicked() {
                    // ファイルを選び、読み込み→デコード→解析を実行してAに格納。
                    if let Some(p) = pick_audio_file() {
                        self.status = format!("Analyzing A: {}", p.display());
                        match analyze_file(&p, self.target_sr, self.analyze_seconds, self.n_fft, self.hop) {
                            Ok(ana) => {
                                self.a = Some(ana);
                                self.status = "A loaded".to_string();
                            }
                            Err(e) => self.status = format!("Error A: {e:#}"),
                        }
                    }
                }
                if ui.button("Load B").clicked() {
                    // ファイルを選び、読み込み→デコード→解析を実行してBに格納。
                    if let Some(p) = pick_audio_file() {
                        self.status = format!("Analyzing B: {}", p.display());
                        match analyze_file(&p, self.target_sr, self.analyze_seconds, self.n_fft, self.hop) {
                            Ok(ana) => {
                                self.b = Some(ana);
                                self.status = "B loaded".to_string();
                            }
                            Err(e) => self.status = format!("Error B: {e:#}"),
                        }
                    }
                }

                ui.separator();

                // フォント（日本語表示などのために、.ttf/.otf を読み込んで egui に適用）
                if ui.button("Load Font").clicked() {
                    if let Some(p) = pick_font_file() {
                        match apply_font_from_file(ctx, &p) {
                            Ok(()) => {
                                if let Err(e) = save_default_font_path(&p) {
                                    self.status = format!("Font applied, but save failed: {e:#}");
                                }
                                self.font_status = format!("Font: {} (default)", p.display());
                            }
                            Err(e) => self.status = format!("Error font: {e:#}"),
                        }
                    }
                }
                if ui.button("Reset Font").clicked() {
                    reset_font(ctx);
                    let _ = clear_default_font_path();
                    self.font_status = "Font: default".to_string();
                }
                ui.label(&self.font_status);

                ui.separator();

                // 解析パラメータ（ロード済みでも変更できるが、反映には Re-analyze が必要）
                ui.label("FFT");
                ui.add(egui::DragValue::new(&mut self.n_fft).range(1024..=131072).speed(1024.0));

                ui.label("Hop");
                ui.add(egui::DragValue::new(&mut self.hop).range(256..=65536).speed(256.0));

                ui.label("Target SR");
                ui.add(egui::DragValue::new(&mut self.target_sr).range(8000..=192000).speed(1000.0));

                ui.label("Analyze sec");
                ui.add(egui::DragValue::new(&mut self.analyze_seconds).range(5.0..=600.0).speed(5.0));

                if ui.button("Re-analyze").clicked() {
                    // 現在ロード済みのパスを保持して、同じファイルを新パラメータで再解析。
                    let a_path = self.a.as_ref().map(|x| x.info.path.clone());
                    let b_path = self.b.as_ref().map(|x| x.info.path.clone());
                    self.a = None;
                    self.b = None;

                    if let Some(s) = a_path {
                        let p = PathBuf::from(s);
                        self.status = format!("Re-analyzing A: {}", p.display());
                        self.a = analyze_file(&p, self.target_sr, self.analyze_seconds, self.n_fft, self.hop).ok();
                    }
                    if let Some(s) = b_path {
                        let p = PathBuf::from(s);
                        self.status = format!("Re-analyzing B: {}", p.display());
                        self.b = analyze_file(&p, self.target_sr, self.analyze_seconds, self.n_fft, self.hop).ok();
                    }
                    self.status = "Re-analyze done".to_string();
                }

                ui.separator();
                ui.label(&self.status);
            });
        });

        // 左: ファイル情報とピーク一覧（A/B）
        egui::SidePanel::left("left").resizable(true).min_width(330.0).show(ctx, |ui| {
            ui.heading("File A");
            if let Some(a) = &self.a {
                show_file_info(ui, &a.info);
                ui.separator();
                ui.label("Top peaks (Hz, dB)");
                show_peaks(ui, &a.peaks);
            } else {
                ui.label("Not loaded");
            }

            ui.separator();
            ui.heading("File B");
            if let Some(b) = &self.b {
                show_file_info(ui, &b.info);
                ui.separator();
                ui.label("Top peaks (Hz, dB)");
                show_peaks(ui, &b.peaks);
            } else {
                ui.label("Not loaded");
            }
        });

        // 中央: 平均スペクトルの比較プロット
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Mean Spectrum (STFT averaged)");
            ui.label("y: dB relative (0 = peak), x: Hz");

            let plot = Plot::new("spectrum_plot")
                .height(ui.available_height())
                .include_x(0.0)
                .include_x(24000.0)
                .include_y(-90.0)
                .include_y(0.0)
                .allow_zoom(true)
                .allow_drag(true);

            plot.show(ui, |plot_ui| {
                if let Some(a) = &self.a {
                    // `egui_plot 0.34` の `Line::new(name, series)` を使用。
                    let line = Line::new("A", PlotPoints::from(a.plot_xy.clone()));
                    plot_ui.line(line);
                }
                if let Some(b) = &self.b {
                    let line = Line::new("B", PlotPoints::from(b.plot_xy.clone()));
                    plot_ui.line(line);
                }
            });
        });
    }
}

fn pick_audio_file() -> Option<PathBuf> {
    // OS標準のファイルピッカー。
    FileDialog::new()
        .add_filter("Audio", &["flac", "mp3", "m4a", "aac", "wav", "ogg", "opus", "webm"])
        .pick_file()
}

fn pick_font_file() -> Option<PathBuf> {
    FileDialog::new().add_filter("Font", &["ttf", "otf"]).pick_file()
}

fn font_config_path() -> Option<PathBuf> {
    // 実行ファイルの隣に、選択したフォントパスを保存する（次回起動時のデフォルト用）。
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|dir| dir.join("audio_analyzer.font_path.txt")))
}

fn save_default_font_path(path: &Path) -> Result<()> {
    let Some(cfg) = font_config_path() else {
        return Err(anyhow!("cannot determine config path"));
    };
    std::fs::write(cfg, path.to_string_lossy().trim()).context("write font config")?;
    Ok(())
}

fn load_default_font_path() -> Option<PathBuf> {
    let cfg = font_config_path()?;
    let text = std::fs::read_to_string(cfg).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(PathBuf::from(trimmed))
}

fn clear_default_font_path() -> Result<()> {
    let Some(cfg) = font_config_path() else {
        return Ok(());
    };
    match std::fs::remove_file(cfg) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).context("remove font config")?,
    }
}

fn find_default_font_candidate() -> Option<PathBuf> {
    // 設定ファイルが無い/空のときのフォールバック。
    // ここにフォントファイルを置けば、最初から日本語フォントを適用できる。
    let candidates = [
        "./meiryo.ttc"
    ];

    let mut bases = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            bases.push(dir.to_path_buf());
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        bases.push(cwd);
    }

    for base in bases {
        for rel in candidates {
            let p = base.join(rel);
            if p.is_file() {
                return Some(p);
            }
        }
    }

    None
}

fn apply_font_from_file(ctx: &egui::Context, path: &Path) -> Result<()> {
    // eguiはシステムフォントを自動では使わないため、必要なフォント（例: NotoSansJP等）を
    // ファイルから読み込んで `FontDefinitions` に登録する。
    let data = std::fs::read(path).with_context(|| format!("read font {}", path.display()))?;

    let mut fonts = egui::FontDefinitions::default();
    fonts
        .font_data
        .insert("user_font".to_string(), egui::FontData::from_owned(data).into());

    // Proportional/Monospace どちらも最優先で user_font を使う。
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "user_font".to_string());
    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .insert(0, "user_font".to_string());

    ctx.set_fonts(fonts);
    ctx.request_repaint();
    Ok(())
}

fn reset_font(ctx: &egui::Context) {
    ctx.set_fonts(egui::FontDefinitions::default());
    ctx.request_repaint();
}

fn show_file_info(ui: &mut egui::Ui, info: &FileInfo) {
    // 1行ずつ等幅フォントで表示して、桁が揃うようにする。
    ui.monospace(format!("Path: {}", info.path));
    ui.monospace(format!("Size: {:.2} MB", info.size_mb));
    ui.monospace(format!("Container: {}", info.container));
    ui.monospace(format!("Codec: {}", info.codec));
    ui.monospace(format!("SR: {} Hz", info.sample_rate));
    ui.monospace(format!("Channels: {}", info.channels));
    ui.monospace(format!("Duration: {:.3} s", info.duration_s));
    if let Some(b) = info.bitrate_kbps {
        ui.monospace(format!("Bitrate: {} kbps", b));
    } else {
        ui.monospace("Bitrate: (unknown)");
    }
    if !info.tags.is_empty() {
        ui.separator();
        ui.label("Tags");
        for (k, v) in &info.tags {
            ui.monospace(format!("{k}: {v}"));
        }
    }
}

fn show_peaks(ui: &mut egui::Ui, peaks: &[(f32, f32)]) {
    // スクロール可能な簡易表。
    egui::ScrollArea::vertical().max_height(160.0).show(ui, |ui| {
        for (hz, db) in peaks.iter().take(12) {
            ui.monospace(format!("{:8.1} Hz   {:6.1} dB", hz, db));
        }
    });
}

fn analyze_file(path: &Path, target_sr: u32, seconds: f64, n_fft: usize, hop: usize) -> Result<Analysis> {
    // 1) デコードして f32 モノラルへ（必要なら target_sr にリサンプル）
    let (info, mono_f32) = decode_to_mono_f32(path, target_sr, seconds)?;
    // 2) STFT平均スペクトルを相対dBで算出
    let (freqs, mean_db) = mean_spectrum_db(&mono_f32, target_sr as f32, n_fft, hop)?;

    // 3) 表示用に 0..24kHz（target_sr=48k想定）に制限して点列化（描画を軽くする）
    let mut plot_xy: Vec<[f64; 2]> = Vec::new();
    for (f, db) in freqs.iter().zip(mean_db.iter()) {
        if *f <= 24000.0 {
            plot_xy.push([*f as f64, *db as f64]);
        }
    }

    // 4) “音の特徴”として分かりやすいピークを抽出（単純な局所最大）
    let peaks = find_peaks(&freqs, &mean_db, 12, 50.0 /*min distance Hz*/);

    Ok(Analysis {
        info,
        freqs_hz: freqs,
        mean_db: mean_db.clone(),
        peaks,
        plot_xy,
    })
}

fn decode_to_mono_f32(path: &Path, target_sr: u32, seconds: f64) -> Result<(FileInfo, Vec<f32>)> {
    // 指定ファイルをデコードして「f32モノラル」に変換し、必要なら `target_sr` にリサンプルして返す。
    //
    // 設計方針:
    // - 比較のためにチャンネルは平均してモノラルへ落とす（L+R+... / Nch）
    // - 解析量を制限するため `seconds` 分相当のサンプル数で打ち切る
    // - サンプルレートが異なる音源同士も比較できるよう `target_sr` へ統一する
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let size_mb = file.metadata().ok().map(|m| m.len() as f64 / 1024.0 / 1024.0).unwrap_or(0.0);

    // Symphoniaは `MediaSourceStream` を介して読み込む。
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|x| x.to_str()) {
        // 拡張子ヒントがあるとフォーマット判定が安定しやすい。
        hint.with_extension(ext);
    }

    // コンテナ(デマルチプレクサ)を推定して `FormatReader` を作る。
    let probed = get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;

    // メタデータ（タグ）
    let mut tags = Vec::new();
    if let Some(meta) = format.metadata().current() {
        for t in meta.tags() {
            tags.push((t.key.to_string(), t.value.to_string()));
        }
    }

    // 通常はデフォルトトラック（多くの場合、音声）を対象にする。
    let track = format
        .default_track()
        .ok_or_else(|| anyhow!("no default track"))?
        .clone();

    let codec = track.codec_params.codec.to_string();
    // サンプルレートはコンテナのヘッダから取れることが多いが、取れない場合もある。
    // 取れないときは後で `decoded.spec().rate` から推定する。
    let sample_rate_from_params = track.codec_params.sample_rate;
    let mut input_sr = sample_rate_from_params.unwrap_or(target_sr);
    let _channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(2);

    // duration
    // time_base と n_frames が揃っていれば、全体の秒数を計算できる。
    let duration_s = if let (Some(n_frames), Some(tb)) = (track.codec_params.n_frames, track.codec_params.time_base) {
        tb.calc_time(n_frames).seconds as f64 + tb.calc_time(n_frames).frac as f64
    } else {
        0.0
    };

    // コーデックデコーダを作成（MP3/FLAC等は Cargo.toml の feature で有効化されている必要がある）。
    let mut decoder = get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    // 表示用の「コンテナ名」。ここでは簡易的に拡張子を入れている。
    let container = path
        .extension()
        .and_then(|x| x.to_str())
        .unwrap_or("unknown")
        .to_string();

    // できるだけ "seconds" 分だけ読む
    // 出力は最終的に `target_sr` に揃えるので、上限サンプル数も `target_sr` 基準で計算する。
    let max_samples = (seconds * target_sr as f64) as usize;
    let mut out: Vec<f32> = Vec::with_capacity(max_samples.min(10_000_000));

    // resampler（必要なら）
    // `FftFixedIn` は入力フレーム数が固定（input_frames_next()）であることが前提。
    // MP3 はデコード毎のフレーム数が 1024/1152 など一定にならないことがあるため、
    // 後段で「pending に貯めて固定長で取り出す」処理を行う。
    let mut resampler: Option<FftFixedIn<f32>> = if input_sr != target_sr {
        Some(FftFixedIn::<f32>::new(input_sr as usize, target_sr as usize, 1024, 2, 1)?)
    } else {
        None
    };
    let mut resample_chunk_in = resampler.as_ref().map(|r| r.input_frames_next()).unwrap_or(0);
    if resampler.is_some() && resample_chunk_in == 0 {
        return Err(anyhow!("resampler input chunk size is 0"));
    }
    // リサンプル用の入力バッファ（モノラルf32）。
    // `pending_offset` は「消費済みの先頭位置」。大きくなったら drain で詰める。
    let mut pending: Vec<f32> = Vec::new();
    let mut pending_offset = 0usize;

    loop {
        if out.len() >= max_samples {
            break;
        }

        // 次のパケットを読む（EOF/エラーで終了）。
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break,
        };

        // デフォルトトラック以外（例: ビデオ/別音声/字幕）が混ざっていても無視する。
        if packet.track_id() != track.id {
            continue;
        }

        // パケットをデコード（失敗するパケットはスキップして継続）。
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(_) => continue,
        };

        // コンテナ側で sample_rate が取れなかった場合は、最初にデコードできた情報から推定する。
        // もし推定値が変化したら、リサンプラを作り直して pending もリセットする。
        if sample_rate_from_params.is_none() {
            let detected_sr = decoded.spec().rate;
            if detected_sr != input_sr {
                input_sr = detected_sr;
                resampler = if input_sr != target_sr {
                    Some(FftFixedIn::<f32>::new(input_sr as usize, target_sr as usize, 1024, 2, 1)?)
                } else {
                    None
                };
                resample_chunk_in = resampler.as_ref().map(|r| r.input_frames_next()).unwrap_or(0);
                if resampler.is_some() && resample_chunk_in == 0 {
                    return Err(anyhow!("resampler input chunk size is 0"));
                }
                pending.clear();
                pending_offset = 0;
            }
        }

        // Symphoniaの任意型バッファを f32 モノへ変換。
        let mut mono_block = audio_to_mono_f32(decoded)?;

        if let Some(r) = resampler.as_mut() {
            // 固定長入力が必要なので、まず pending へ貯める。
            pending.extend_from_slice(&mono_block);

            // pending が十分に溜まっている間、固定長で切り出してリサンプルする。
            while pending.len().saturating_sub(pending_offset) >= resample_chunk_in && out.len() < max_samples {
                let in_slice = &pending[pending_offset..(pending_offset + resample_chunk_in)];
                let out_blocks = r.process(&[in_slice], None)?;
                if let Some(ch0) = out_blocks.first() {
                    out.extend_from_slice(ch0);
                }
                pending_offset += resample_chunk_in;
            }

            // 先頭が巨大化してきたら詰める（頻繁なdrainは遅いので閾値を設ける）。
            if pending_offset >= 65_536 {
                pending.drain(..pending_offset);
                pending_offset = 0;
            }
        } else {
            // リサンプル不要なら、そのまま出力へ追加。
            out.append(&mut mono_block);
        }
    }

    if let Some(r) = resampler.as_mut() {
        if out.len() < max_samples {
            // ストリーム末尾の端数は `process_partial` でゼロ詰め相当として吐き出す。
            let remaining = &pending[pending_offset..];
            if !remaining.is_empty() {
                let out_blocks = r.process_partial(Some(&[remaining]), None)?;
                if let Some(ch0) = out_blocks.first() {
                    out.extend_from_slice(ch0);
                }
            }
        }
    }

    // 指定秒数ぶんで切り詰め。
    out.truncate(max_samples);

    let info = FileInfo {
        path: path.to_string_lossy().to_string(),
        container,
        codec,
        sample_rate: target_sr,
        channels: 1,
        duration_s,
        bitrate_kbps: None,
        tags,
        size_mb,
    };

    Ok((info, out))
}

fn audio_to_mono_f32(decoded: AudioBufferRef<'_>) -> Result<Vec<f32>> {
    // Symphoniaの `AudioBufferRef` は型が様々（u8/i16/f32...）なので、いったん f32 へ変換して統一する。
    // `convert` は内部で適切なスケーリングを行うため、型ごとの分岐を自前で書かなくてよい。
    let mut buf_f32: AudioBuffer<f32> = decoded.make_equivalent::<f32>();
    decoded.convert(&mut buf_f32);
    mono_from_f32(&buf_f32)
}

fn mono_from_f32(buf: &AudioBuffer<f32>) -> Result<Vec<f32>> {
    // 非インターリーブ（chごとの平面）を、フレーム単位で平均してモノラルへ。
    let chans = buf.spec().channels.count();
    let frames = buf.frames();
    let mut mono = Vec::with_capacity(frames);
    for i in 0..frames {
        let mut s = 0.0f32;
        for ch in 0..chans {
            s += buf.chan(ch)[i];
        }
        mono.push(s / chans as f32);
    }
    Ok(mono)
}

fn mean_spectrum_db(x: &[f32], sr: f32, n_fft: usize, hop: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    // 連続波形 `x` から STFT を作り、各周波数ビンの平均振幅を求める。
    // - 窓: Hann
    // - 振幅スペクトルをフレーム方向に平均
    // - 最大値を 0dB に正規化した相対dB（比較用途）
    if x.len() < n_fft {
        return Err(anyhow!("audio too short for FFT (need >= n_fft samples)"));
    }

    // FFTプランは使い回す。
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    // Hann窓を事前計算。
    let mut window = vec![0.0f32; n_fft];
    for i in 0..n_fft {
        // Hann
        window[i] = 0.5 - 0.5 * ((2.0 * std::f32::consts::PI * i as f32) / (n_fft as f32)).cos();
    }

    // 実数入力のFFT結果は対称になるので、正の周波数側（DC〜Nyquist）のみ使う。
    let n_bins = n_fft / 2 + 1;
    let mut acc = vec![0.0f32; n_bins];
    let mut frames = 0usize;

    let mut buf = vec![Complex32::new(0.0, 0.0); n_fft];

    let mut pos = 0usize;
    while pos + n_fft <= x.len() {
        // 窓掛けして複素バッファへ。
        for i in 0..n_fft {
            buf[i].re = x[pos + i] * window[i];
            buf[i].im = 0.0;
        }

        // FFT実行（インプレース）。
        fft.process(&mut buf);

        // 振幅（|X|）を加算して平均用に蓄積。
        for k in 0..n_bins {
            let mag = (buf[k].re * buf[k].re + buf[k].im * buf[k].im).sqrt();
            acc[k] += mag;
        }

        frames += 1;
        pos += hop;
    }

    if frames == 0 {
        return Err(anyhow!("no frames"));
    }

    // 平均化 + 0除算回避のための下限。
    for v in acc.iter_mut() {
        *v /= frames as f32;
        if *v < 1e-12 {
            *v = 1e-12;
        }
    }

    // 相対dB化の基準（最大=0dB）。
    let max = acc.iter().cloned().fold(0.0f32, f32::max).max(1e-12);

    let mut freqs = Vec::with_capacity(n_bins);
    let mut db = Vec::with_capacity(n_bins);
    for k in 0..n_bins {
        // 周波数軸（Hz）と相対dB。
        let f = (k as f32) * sr / (n_fft as f32);
        let rel = acc[k] / max;
        let d = 20.0 * rel.log10();
        freqs.push(f);
        db.push(d);
    }

    Ok((freqs, db))
}

fn find_peaks(freqs: &[f32], db: &[f32], top_n: usize, min_dist_hz: f32) -> Vec<(f32, f32)> {
    // 単純な局所最大ピーク検出 + 距離制約
    let mut candidates = Vec::<(f32, f32)>::new();

    // 近傍2点と比較するだけの簡易ピーク（ノイズに弱いが、表示用途としては軽い）。
    for i in 1..(db.len().saturating_sub(1)) {
        if db[i] > db[i - 1] && db[i] > db[i + 1] {
            candidates.push((freqs[i], db[i]));
        }
    }

    // dBが高い順
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut picked: Vec<(f32, f32)> = Vec::new();
    'outer: for (f, d) in candidates {
        // 近接したピークを除外（例: 同じピークの肩を複数拾わないため）。
        for (pf, _) in &picked {
            if (f - *pf).abs() < min_dist_hz {
                continue 'outer;
            }
        }
        picked.push((f, d));
        if picked.len() >= top_n {
            break;
        }
    }

    picked
}
