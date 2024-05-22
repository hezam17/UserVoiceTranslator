import 'package:permission_handler/permission_handler.dart';

class PermissionsUtil {
  static Future<void> requestPermissions() async {
    await [Permission.microphone, Permission.storage].request();
  }
}
