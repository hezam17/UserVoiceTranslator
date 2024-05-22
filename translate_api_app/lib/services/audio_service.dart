import 'package:flutter_sound/flutter_sound.dart';
import '../models/audio_model.dart';
import '../utils/permissions_util.dart';

class AudioService {
  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  final FlutterSoundPlayer _player = FlutterSoundPlayer();
  bool _isRecording = false;
  bool _isPlaying = false;

  bool get isRecording => _recorder.isRecording;
  bool get isPlaying => _player.isPlaying;

  Future<void> init() async {
    await PermissionsUtil.requestPermissions();
    await _recorder.openRecorder();
    await _player.openPlayer();
    _recorder.setSubscriptionDuration(const Duration(milliseconds: 500));
  }

  Future<void> startRecording() async {
    _isRecording = true;
    await _recorder.startRecorder(toFile: 'path_to_file');
  }

  Future<AudioModel> stopRecording() async {
    _isRecording = false;
    String? path = await _recorder.stopRecorder();
    return AudioModel(path: path);
  }

  Future<void> startPlaying(String? path) async {
    if (path != null) {
      _isPlaying = true;
      await _player.startPlayer(fromURI: path);
    } else {
      print("No valid path provided for playing.");
    }
  }

  Future<void> stopPlaying() async {
    _isPlaying = false;
    await _player.stopPlayer();
  }

  void dispose() {
    _recorder.closeRecorder();
    _player.closePlayer();
  }
}
