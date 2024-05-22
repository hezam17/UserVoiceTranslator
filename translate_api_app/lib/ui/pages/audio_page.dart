import 'package:flutter/material.dart';
import '../../services/audio_service.dart';
import '../../models/audio_model.dart';

class AudioPage extends StatefulWidget {
  @override
  _AudioPageState createState() => _AudioPageState();
}

class _AudioPageState extends State<AudioPage> {
  final AudioService _audioService = AudioService();
  AudioModel? _currentRecording;

  String _selectedSourceLanguage = 'English';
  String _selectedTargetLanguage = 'Arabic';
  List<String> _languages = [
    'English',
    'Arabic',
    'Russian',
    'Spanish',
  ];
  List<String> _languagesCode = [
    'en',
    'ar',
    'ru',
    'es',
  ];

  @override
  void initState() {
    super.initState();
    _audioService.init().catchError((error) {
      ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Initialization failed: $error")));
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: const Text('Translator App',
            style: TextStyle(fontWeight: FontWeight.bold, fontSize: 28)),
      ),
      backgroundColor: Colors.grey[200],
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text('From: '),
                DropdownButton<String>(
                  value: _selectedSourceLanguage,
                  items: _languages.map((String value) {
                    return DropdownMenuItem<String>(
                      value: value,
                      child: Text(value),
                    );
                  }).toList(),
                  onChanged: (String? newValue) {
                    setState(() {
                      _selectedSourceLanguage = newValue!;
                    });
                  },
                ),
                SizedBox(width: 20),
                Text('To: '),
                DropdownButton<String>(
                  value: _selectedTargetLanguage,
                  items: _languages.map((String value) {
                    return DropdownMenuItem<String>(
                      value: value,
                      child: Text(value),
                    );
                  }).toList(),
                  onChanged: (String? newValue) {
                    setState(() {
                      _selectedTargetLanguage = newValue!;
                    });
                  },
                ),
              ],
            ),
            ElevatedButton(
              onPressed:
                  _audioService.isRecording ? _stopRecording : _startRecording,
              child: Text(_audioService.isRecording
                  ? 'Stop Recording'
                  : 'Start Recording'),
            ),
            if (_currentRecording?.path != null)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 8.0),
                child: Text('Recording Path: ${_currentRecording!.path}'),
              ),
            ElevatedButton(
              onPressed: _audioService.isPlaying ? _stopPlaying : _startPlaying,
              child: Text(
                  _audioService.isPlaying ? 'Stop Playing' : 'Start Playing'),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _startRecording() async {
    await _audioService.startRecording();
    setState(() {});
  }

  Future<void> _stopRecording() async {
    _currentRecording = await _audioService.stopRecording();
    setState(() {});
  }

  Future<void> _startPlaying() async {
    if (_currentRecording?.path != null) {
      await _audioService.startPlaying(_currentRecording!.path!);
      setState(() {});
    } else {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text(
              "No recording available to play. Please record something first.")));
    }
  }

  Future<void> _stopPlaying() async {
    await _audioService.stopPlaying();
    setState(() {});
  }

  @override
  void dispose() {
    _audioService.dispose();
    super.dispose();
  }
}
