import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

class ApiService {
  static Future<Map<String, dynamic>> sendFile(
      String filePath, String sourceLanguage, String targetLanguage) async {
    try {
      var uri = Uri.parse('http://10.0.2.2:5000/process_clip');
      var request = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('audio_clip', filePath))
        ..fields['source_language'] = sourceLanguage
        ..fields['target_language'] = targetLanguage;

      var response = await request.send();
      if (response.statusCode == 200) {
        String responseData = await response.stream.bytesToString();
        return json.decode(responseData);
      } else {
        throw Exception('Failed to upload. Status: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Failed to connect: $e');
    }
  }
}
