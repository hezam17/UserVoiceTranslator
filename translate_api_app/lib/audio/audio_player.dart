import 'dart:typed_data';

import 'package:flutter_sound/flutter_sound.dart';

class AudioPlayer {
  FlutterSoundPlayer? _player;

  AudioPlayer() {
    _player = FlutterSoundPlayer();
  }

  Future<void> startPlaying(String path) async {
    await _player!.openPlayer();
    await _player!.startPlayer(fromURI: path, codec: Codec.pcm16WAV);
  }

  Future<void> stopPlaying() async {
    await _player!.stopPlayer();
    await _player!.closePlayer();
  }

  Future<void> playFromBytes(Uint8List bytes) async {
    await _player!.startPlayer(
      codec: Codec.pcm16WAV,
      fromDataBuffer: bytes,
      whenFinished: () {
        print("Finished playing");
      },
    );
  }
}
