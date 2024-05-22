import 'dart:io';

import 'package:flutter_sound/flutter_sound.dart';
import 'package:path_provider/path_provider.dart';

class AudioRecorder {
  FlutterSoundRecorder? _recorder;

  AudioRecorder() {
    _recorder = FlutterSoundRecorder();
  }

  Future<String> startRecording() async {
    final directory = await getApplicationDocumentsDirectory();
    final path = directory.path + '/flutter_sound_record.wav';

    await _recorder!.openRecorder();
    _recorder!.setSubscriptionDuration(const Duration(milliseconds: 500));
    await _recorder!.startRecorder(toFile: path, codec: Codec.pcm16WAV);

    return path;
  }

  Future<void> stopRecording() async {
    await _recorder!.stopRecorder();
    await _recorder!.closeRecorder();
  }
}
