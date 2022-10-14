package com.example.tfltest;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.BatteryManager;
import android.os.Bundle;
import com.google.android.material.snackbar.Snackbar;

import androidx.appcompat.app.AppCompatActivity;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import com.example.tfltest.databinding.ActivityMainBinding;
import com.google.mediapipe.formats.proto.LandmarkProto;
import com.google.mediapipe.solutions.hands.HandLandmark;
import com.google.mediapipe.solutions.hands.Hands;
import com.google.mediapipe.solutions.hands.HandsOptions;
import com.google.mediapipe.solutions.hands.HandsResult;
import com.google.protobuf.ByteString;

import edu.cmu.cs.gabriel.client.comm.ServerComm;
import edu.cmu.cs.gabriel.client.results.ErrorType;
import edu.cmu.cs.gabriel.protocol.Protos.InputFrame;
import edu.cmu.cs.gabriel.protocol.Protos.ResultWrapper;
import edu.cmu.cs.gabriel.protocol.Protos.PayloadType;

import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;


public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private static final String SOURCE = "profiling";
    private static final int PORT = 9099;

    private static final String LOGFILE = "TFLTest.txt";
    private static final long TIMER_PERIOD = 1000;

    private AppBarConfiguration appBarConfiguration;
    private ActivityMainBinding binding;
    private ExecutorService pool;
    private BatteryManager mBatteryManager;
    private BroadcastReceiver batteryReceiver;
    private FileWriter logFileWriter;
    private Timer timer;
    ConcurrentLinkedDeque<String> logList;

    private ServerComm serverComm;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // UI
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setSupportActionBar(binding.toolbar);

        NavController navController = Navigation.findNavController(this,
                R.id.nav_host_fragment_content_main);
        appBarConfiguration = new AppBarConfiguration.Builder(navController.getGraph()).build();
        NavigationUI.setupActionBarWithNavController(this, navController, appBarConfiguration);

        binding.fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
                        .setAction("Action", null).show();
            }
        });

        // Server Communication
        Consumer<ResultWrapper> consumer = resultWrapper -> {
            if (resultWrapper.getResultsCount() == 0) {
                return;
            }

            ResultWrapper.Result result = resultWrapper.getResults(0);
            ByteString jpegByteString = result.getPayload();

            // TODO: Handle payload

        };

        Consumer<ErrorType> onDisconnect = errorType -> {
            Log.e(TAG, "Disconnect Error:" + errorType.name());
            finish();
        };

        serverComm = ServerComm.createServerComm(
                consumer, BuildConfig.GABRIEL_HOST, PORT, getApplication(), onDisconnect);

        // Profiling tests
        File logFile = new File(getExternalFilesDir(null), LOGFILE);
        logFile.delete();
        logFile = new File(getExternalFilesDir(null), LOGFILE);
//        Log.d(TAG, logFile.getPath());
        try {
            logFileWriter = new FileWriter(logFile, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
        logList = new ConcurrentLinkedDeque<>();

        IntentFilter intentFilter = new IntentFilter(Intent.ACTION_BATTERY_CHANGED);
        batteryReceiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                int voltage = intent.getIntExtra(BatteryManager.EXTRA_VOLTAGE, Integer.MIN_VALUE);
                String voltageMsg = TAG + ": Time: " + SystemClock.uptimeMillis() +
                        "\tBattery voltage = " + voltage + "\n";
                logList.add(voltageMsg);
            }
        };
        registerReceiver(batteryReceiver, intentFilter);

        mBatteryManager = (BatteryManager) getSystemService(Context.BATTERY_SERVICE);

        timer = new Timer();
        timer.scheduleAtFixedRate(new LogTimerTask(), 0, TIMER_PERIOD);
        pool = Executors.newFixedThreadPool(1);

        // 1. Test thumbs-up detection
//        pool.execute(this::testThumbsUp);

        // 2. Test pipeline
        pool.execute(() -> {
            Pipeline pipeline = new Pipeline(this);
//            pipeline.testPipeline();
            pipeline.testPHashPipeline();
            writeLog();
        });

        // 3. Test perceptual hashing
//        pool.execute(() -> {
//            ImagePHash imagePHash = new ImagePHash(this);
//            imagePHash.testPHash();
//            writeLog();
//        });
    }

    class LogTimerTask extends TimerTask {
        @Override
        public void run() {
            int current = mBatteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW);
            String testMag = TAG + ": Time: " + SystemClock.uptimeMillis() +
                    "\tCurrent: " + current + "\n";
            logList.add(testMag);
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    public boolean onSupportNavigateUp() {
        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment_content_main);
        return NavigationUI.navigateUp(navController, appBarConfiguration)
                || super.onSupportNavigateUp();
    }

    private Hands hands;

    private void writeLog() {
        timer.cancel();
        timer.purge();
        unregisterReceiver(batteryReceiver);
        try {
            for (String logString: logList) {
                logFileWriter.write(logString);
            }
            logFileWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        runOnUiThread(() -> {
            binding.fab.setVisibility(View.INVISIBLE);
        });
        Log.d(TAG, "Profiling completed.");
    }

    public void testThumbsUp() {
        final String testBasedir = Environment.getExternalStorageDirectory().getPath() +
                "/test_images/hands";
        File testImages = new File(testBasedir + "/unknown");
        final int imageCount = testImages.list().length;
//        Log.d(TAG, "Total Images: " + imageCount);

        final int MAX_TOKEN = 2;
        AtomicInteger flowControlToken = new AtomicInteger(MAX_TOKEN);

        AtomicInteger good = new AtomicInteger();
        AtomicInteger bad = new AtomicInteger();
        AtomicInteger thumbsUp = new AtomicInteger();
        AtomicInteger processed = new AtomicInteger();

        HandsOptions handsOptions = HandsOptions.builder()
                .setStaticImageMode(true)
                .setMaxNumHands(2)
                .setRunOnGpu(true)
                .build();
        hands = new Hands(this, handsOptions);

        hands.setResultListener(
                handsResult -> {
                    processed.incrementAndGet();
                    if (handsResult.multiHandLandmarks().isEmpty()) {
                        bad.incrementAndGet();
                    } else {
                        good.incrementAndGet();
                        if (detectThumbsUp(handsResult)) {
                            thumbsUp.incrementAndGet();
                        }
                    }
                    flowControlToken.incrementAndGet();
                });
        hands.setErrorListener((message, e) -> Log.e(TAG, "MediaPipe Hands error:" + message));

//        detectHands(testBasedir + "/2022-09-02-22-03-36-685853-none(bolt).jpg");
//        Log.d(TAG, "Warmup finished.");

        File[] imageFiles = testImages.listFiles();
        logList.add(TAG + ": Start: " + SystemClock.uptimeMillis() + "\n");

        for (int i = 0; i < imageCount; i++) {
            File imageFile = imageFiles[i];
            while (flowControlToken.get() <= 0) {
            }
            detectHands(imageFile.getPath());
            flowControlToken.decrementAndGet();
        }
        while (flowControlToken.get() < MAX_TOKEN) {
        }

        logList.add(TAG + ": Total Images: " + processed.get() + "\n");
        logList.add(TAG + ": Stop: " + SystemClock.uptimeMillis() + "\n");

        writeLog();
    }

    private void detectHands(String imagePath) {
        Bitmap image = BitmapFactory.decodeFile(imagePath);
        if (image != null) {
            hands.send(image);
        }
    }

    private boolean detectThumbsUp(HandsResult result) {
        HashMap<String, Object> handState = getHandState(result);
        if ((Boolean)handState.get("thumb_open") &&
                (Boolean)handState.get("index_finger_closed") &&
                (Boolean)handState.get("middle_finger_closed") &&
                (Boolean)handState.get("ring_finger_closed") &&
                (Boolean)handState.get("pinky_closed")) {

            if ((Double)handState.get("thumb_middle_angle") <= 90 &&
                    (Double)handState.get("thumb_middle_angle") > (Double)handState.get("thumb_index_angle") &&
                    (Double)handState.get("thumb_index_angle") > 15) {

                if ((Double)handState.get("orientation") > 120 ||
                        (Double)handState.get("orientation") < -150 ||
                        ((Double)handState.get("orientation") < 60 &&
                                (Double)handState.get("orientation") > -30)) {

                    if (handState.get("thumb_orientation").equals("up") &&
                            handState.get("finger_y_order").equals("up")) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    private double dist2D(double x1, double y1, double x2, double y2) {
        return Math.hypot(Math.abs(y2 - y1), Math.abs(x2 - x1));
    }

    private double dot2D(double v1x, double v1y, double v2x, double v2y) {
        return v1x * v2x + v1y * v2y;
    }

    private double vectorAngleDegree(double v1x, double v1y, double v2x, double v2y) {
        return Math.toDegrees(Math.acos(dot2D(v1x, v1y, v2x, v2y) /
                Math.sqrt(dot2D(v1x, v1y, v1x, v1y)) /
                Math.sqrt(dot2D(v2x, v2y, v2x, v2y))));
    }

    private HashMap<String, Object> getHandState(HandsResult result) {
        int width = result.inputBitmap().getWidth();
        int height = result.inputBitmap().getHeight();
        List<LandmarkProto.NormalizedLandmark> handLandmark =
                result.multiHandLandmarks().get(0).getLandmarkList();

        HashMap<String, Object> handState = new HashMap<>();

        double x0 = handLandmark.get(HandLandmark.WRIST).getX() * width;
        double y0 = handLandmark.get(HandLandmark.WRIST).getY() * height;
        double x9 = handLandmark.get(HandLandmark.MIDDLE_FINGER_MCP).getX() * width;
        double y9 = handLandmark.get(HandLandmark.MIDDLE_FINGER_MCP).getY() * height;
        handState.put("orientation", Math.toDegrees(Math.atan2(y0 - y9, x9 - x0)));

        double x1 = handLandmark.get(HandLandmark.THUMB_CMC).getX() * width;
        double y1 = handLandmark.get(HandLandmark.THUMB_CMC).getY() * height;
        double x2 = handLandmark.get(HandLandmark.THUMB_MCP).getX() * width;
        double y2 = handLandmark.get(HandLandmark.THUMB_MCP).getY() * height;
        double x3 = handLandmark.get(HandLandmark.THUMB_IP).getX() * width;
        double y3 = handLandmark.get(HandLandmark.THUMB_IP).getY() * height;
        double x4 = handLandmark.get(HandLandmark.THUMB_TIP).getX() * width;
        double y4 = handLandmark.get(HandLandmark.THUMB_TIP).getY() * height;
        double d01 = dist2D(x0, y0, x1, y1);
        double d02 = dist2D(x0, y0, x2, y2);
        double d03 = dist2D(x0, y0, x3, y3);
        double d04 = dist2D(x0, y0, x4, y4);
        handState.put("thumb_open", d04 > d03 && d03 > d02 && d02 > d01);

        double x5 = handLandmark.get(HandLandmark.INDEX_FINGER_MCP).getX() * width;
        double y5 = handLandmark.get(HandLandmark.INDEX_FINGER_MCP).getY() * height;
        double v04x = x4 - x0;
        double v04y = y4 - y0;
        double v05x = x5 - x0;
        double v05y = y5 - y0;
        double v09x = x9 - x0;
        double v09y = y9 - y0;
        handState.put("thumb_index_angle", vectorAngleDegree(v04x, v04y, v05x, v05y));
        handState.put("thumb_middle_angle", vectorAngleDegree(v04x, v04y, v09x, v09y));

        double x6 = handLandmark.get(HandLandmark.INDEX_FINGER_PIP).getX() * width;
        double y6 = handLandmark.get(HandLandmark.INDEX_FINGER_PIP).getY() * height;
        double x7 = handLandmark.get(HandLandmark.INDEX_FINGER_DIP).getX() * width;
        double y7 = handLandmark.get(HandLandmark.INDEX_FINGER_DIP).getY() * height;
        double x8 = handLandmark.get(HandLandmark.INDEX_FINGER_TIP).getX() * width;
        double y8 = handLandmark.get(HandLandmark.INDEX_FINGER_TIP).getY() * height;
        double d06 = dist2D(x0, y0, x6, y6);
        double d07 = dist2D(x0, y0, x7, y7);
        double d08 = dist2D(x0, y0, x8, y8);
        handState.put("index_finger_closed", d06 > d07 && d07 > d08);

        double x10 = handLandmark.get(HandLandmark.MIDDLE_FINGER_PIP).getX() * width;
        double y10 = handLandmark.get(HandLandmark.MIDDLE_FINGER_PIP).getY() * height;
        double x11 = handLandmark.get(HandLandmark.MIDDLE_FINGER_DIP).getX() * width;
        double y11 = handLandmark.get(HandLandmark.MIDDLE_FINGER_DIP).getY() * height;
        double x12 = handLandmark.get(HandLandmark.MIDDLE_FINGER_TIP).getX() * width;
        double y12 = handLandmark.get(HandLandmark.MIDDLE_FINGER_TIP).getY() * height;
        double d010 = dist2D(x0, y0, x10, y10);
        double d011 = dist2D(x0, y0, x11, y11);
        double d012 = dist2D(x0, y0, x12, y12);
        handState.put("middle_finger_closed", d010 > d011 && d011 > d012);

        double x14 = handLandmark.get(HandLandmark.RING_FINGER_PIP).getX() * width;
        double y14 = handLandmark.get(HandLandmark.RING_FINGER_PIP).getY() * height;
        double x15 = handLandmark.get(HandLandmark.RING_FINGER_DIP).getX() * width;
        double y15 = handLandmark.get(HandLandmark.RING_FINGER_DIP).getY() * height;
        double x16 = handLandmark.get(HandLandmark.RING_FINGER_TIP).getX() * width;
        double y16 = handLandmark.get(HandLandmark.RING_FINGER_TIP).getY() * height;
        double d014 = dist2D(x0, y0, x14, y14);
        double d015 = dist2D(x0, y0, x15, y15);
        double d016 = dist2D(x0, y0, x16, y16);
        handState.put("ring_finger_closed", d014 > d015 && d015 > d016);

        double x18 = handLandmark.get(HandLandmark.PINKY_PIP).getX() * width;
        double y18 = handLandmark.get(HandLandmark.PINKY_PIP).getY() * height;
        double x19 = handLandmark.get(HandLandmark.PINKY_DIP).getX() * width;
        double y19 = handLandmark.get(HandLandmark.PINKY_DIP).getY() * height;
        double x20 = handLandmark.get(HandLandmark.PINKY_TIP).getX() * width;
        double y20 = handLandmark.get(HandLandmark.PINKY_TIP).getY() * height;
        double d018 = dist2D(x0, y0, x18, y18);
        double d019 = dist2D(x0, y0, x19, y19);
        double d020 = dist2D(x0, y0, x20, y20);
        handState.put("pinky_closed", d018 > d019 && d019 > d020);

        Integer[] landmarkYOrder = new Integer[handLandmark.size()];
        for (int i = 0; i < landmarkYOrder.length; i++) {
            landmarkYOrder[i] = i;
        }
        // A brief implementation of argsort
        Arrays.sort(landmarkYOrder, (i1, i2) -> Float.compare(handLandmark.get(i1).getY(), handLandmark.get(i2).getY()));
        String thumbOrientation = "unknown";
        if (landmarkYOrder[0] == HandLandmark.THUMB_TIP && landmarkYOrder[1] == HandLandmark.THUMB_IP) {
            thumbOrientation = "up";
        } else if (landmarkYOrder[landmarkYOrder.length - 1] == HandLandmark.THUMB_TIP && landmarkYOrder[landmarkYOrder.length - 2] == HandLandmark.THUMB_IP) {
            thumbOrientation = "down";
        }
        handState.put("thumb_orientation", thumbOrientation);

        String fingerYOrder = "unknown";
        if (y4 < y6 && y6 < y10 && y10 < y14 && y14 < y18) {
            fingerYOrder = "up";
        } else if (y4 > y6 && y6 > y10 && y10 > y14 && y14 > y18) {
            fingerYOrder = "down";
        }
        handState.put("finger_y_order", fingerYOrder);

        return handState;
    }
}