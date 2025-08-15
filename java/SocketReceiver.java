import java.net.*;
import java.io.*;
import org.json.JSONObject;
import org.json.JSONArray;

public class SocketReceiver {
    public static void main(String[] args) {
        int port = 5005;
        try (ServerSocket serverSocket = new ServerSocket(port)) {
            System.out.println("Listening on port " + port + "...");
            
            while (true) { // allow multiple connections
                Socket clientSocket = serverSocket.accept();
                System.out.println("Client connected: " + clientSocket.getInetAddress());

                try (BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()))) {
                    String line;
                    while ((line = in.readLine()) != null) {
                        line = line.trim();
                        if (line.isEmpty()) continue;

                        try {
                            JSONObject obj = new JSONObject(line);
                            String rightHand = obj.optString("right_hand", "None");
                            String leftHand = obj.optString("left_hand", "None");
                            double yaw = obj.optDouble("yaw", 0.0);
                            double pitch = obj.optDouble("pitch", 0.0);

                            System.out.printf("Right: %s | Left: %s | Yaw: %.1f | Pitch: %.1f%n",
                                    rightHand, leftHand, yaw, pitch);

                            if (obj.has("pressed_keys")) {
                                JSONArray keys = obj.getJSONArray("pressed_keys");
                                System.out.print("Pressed keys: ");
                                for (int i = 0; i < keys.length(); i++) {
                                    System.out.print(keys.getString(i));
                                    if (i < keys.length() - 1) System.out.print(", ");
                                }
                                System.out.println();
                            }

                            if (obj.has("pressed_mouse")) {
                                JSONArray mouse = obj.getJSONArray("pressed_mouse");
                                System.out.print("Pressed mouse: ");
                                for (int i = 0; i < mouse.length(); i++) {
                                    System.out.print(mouse.getString(i));
                                    if (i < mouse.length() - 1) System.out.print(", ");
                                }
                                System.out.println();
                            }

                        } catch (Exception e) {
                            System.out.println("Invalid JSON: " + line);
                        }
                    }
                } catch (IOException e) {
                    System.out.println("Client disconnected.");
                } finally {
                    clientSocket.close();
                    System.out.println("Connection closed.");
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
