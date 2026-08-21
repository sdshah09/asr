// Milestone 0: read a WAV file and print what's inside it.

use std::env;

fn main() {
    // Command-line arguments. args() gives an iterator; skip(1) drops the
    // program name; next() takes the first real argument, if there is one.
    let path = env::args().skip(1).next();

    // `path` is an Option<String>: either Some(value) or None. Rust has no
    // null, so "might be missing" is part of the type and must be handled.
    let path = match path {
        Some(p) => p,
        None => {
            eprintln!("usage: dictate <file.wav>");
            return;
        }
    };

    // Opening a file can fail, so this returns a Result: Ok(reader) or Err(e).
    let reader = match hound::WavReader::open(&path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("could not open {path}: {e}");
            return;
        }
    };

    let spec = reader.spec();
    let samples = reader.len();                       // total samples in the file
    let seconds = samples as f64 / spec.sample_rate as f64 / spec.channels as f64;

    println!("file        : {path}");
    println!("sample rate : {} Hz", spec.sample_rate);
    println!("channels    : {}", spec.channels);
    println!("bits        : {}", spec.bits_per_sample);
    println!("samples     : {samples}");
    println!("duration    : {seconds:.2} s");

    // Whisper and Moonshine both want 16 kHz mono.
    if spec.sample_rate != 16_000 || spec.channels != 1 {
        println!("\nnote: needs converting to 16 kHz mono before transcription");
    } else {
        println!("\nready for transcription (already 16 kHz mono)");
    }
}
