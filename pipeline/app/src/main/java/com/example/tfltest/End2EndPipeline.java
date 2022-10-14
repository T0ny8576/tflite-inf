package com.example.tfltest;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;

import com.google.protobuf.ByteString;

import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.core.BaseOptions;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.function.Consumer;

import edu.cmu.cs.gabriel.client.comm.ServerComm;
import edu.cmu.cs.gabriel.client.results.ErrorType;
import edu.cmu.cs.gabriel.protocol.Protos;
import edu.cmu.cs.gabriel.protocol.Protos.InputFrame;
import edu.cmu.cs.gabriel.protocol.Protos.PayloadType;

public class End2EndPipeline {
    private static final String TAG = "Pipeline";
    private static final String SOURCE = "profiling";
    private static final int PORT = 9099;
    ServerComm serverComm;

    ObjectDetector objectDetector = null;
    ConcurrentLinkedDeque<String> logList;
    final boolean OD_ON_GLASS = false;

    public End2EndPipeline(MainActivity mainActivity) {
        this.logList = mainActivity.logList;

        // Server Communication
        Consumer<Protos.ResultWrapper> consumer = resultWrapper -> {
            if (resultWrapper.getResultsCount() == 0) {
                return;
            }

            Protos.ResultWrapper.Result result = resultWrapper.getResults(0);
            ByteString jpegByteString = result.getPayload();

            // TODO: Handle payload

        };

        Consumer<ErrorType> onDisconnect = errorType -> {
            Log.e(TAG, "Disconnect Error:" + errorType.name());
            mainActivity.finish();
        };

        serverComm = ServerComm.createServerComm(
                consumer, BuildConfig.GABRIEL_HOST, PORT, mainActivity.getApplication(), onDisconnect);

        if (OD_ON_GLASS) {
            try {
                ObjectDetector.ObjectDetectorOptions.Builder optionsBuilder =
                        ObjectDetector.ObjectDetectorOptions.builder()
                                .setScoreThreshold(0.4f)
                                .setMaxResults(1);

                BaseOptions.Builder baseOptionsBuilder = BaseOptions.builder().setNumThreads(1);

                // This check is necessary because baseOptionsBuilder.useGpu(); will not work on Google
                // glass.
                if ((new CompatibilityList()).isDelegateSupportedOnThisDevice()) {
                    baseOptionsBuilder.useGpu();
                }

                optionsBuilder.setBaseOptions(baseOptionsBuilder.build());

                File modelFile = new File("/sdcard/tflite_models/ed0.tflite");
                objectDetector = ObjectDetector.createFromFileAndOptions(
                        modelFile, optionsBuilder.build());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private ByteString runOD(Bitmap image) {
        ImageProcessor imageProcessor = (new ImageProcessor.Builder()).build();
        TensorImage tensorImage = imageProcessor.process(TensorImage.fromBitmap(image));

        List<Detection> detections = objectDetector.detect(tensorImage);
        if (detections.size() == 0) {
            return null;
        }
        if (detections.size() > 1) {
            throw new RuntimeException();
        }
        RectF rectF = detections.get(0).getBoundingBox();

        if ((rectF.bottom < 0) || (rectF.top < 0) || (rectF.width() == 0) ||
                (rectF.height() == 0)) {
            return null;
        }

        image = Bitmap.createBitmap(image, (int) rectF.left, (int) rectF.top,
                (int) rectF.width(), (int) rectF.height());

        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        image.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
        return ByteString.copyFrom(byteArrayOutputStream.toByteArray());
    }

    public void testPipeline() {
        if (OD_ON_GLASS) {
            runOD(BitmapFactory.decodeFile("/sdcard/test_images/stirling/1screw/2_frame-0000.jpg"));
            Log.d(TAG, "Warmup finished.");
        }

        File testImages = new File( Environment.getExternalStorageDirectory().getPath()
                + "/test_images/stirling");
        final int folderCount = testImages.list().length;
        final int testImagePerFolder = 150;
        final int imageCount = testImagePerFolder * folderCount;
        File[] classDirs = testImages.listFiles();

        logList.add(TAG + ": Total Images: " + imageCount + "\n");
        logList.add(TAG + ": Start: " + SystemClock.uptimeMillis() + "\n");

        for (int i = 0; i < folderCount; i++) {
            File classDir = classDirs[i];
            String correctClass = classDir.getName();
//            final int imageCount = classDir.list().length;
//            Log.d(TAG, "Total Images: " + imageCount);

            File[] imageFiles = classDir.listFiles();
            for (int j = 0; j < testImagePerFolder; j++) {
                File imageFile = imageFiles[j];
                Bitmap image = BitmapFactory.decodeFile(imageFile.getPath());

                if (OD_ON_GLASS) {
                    ByteString croppedByteString = runOD(image);
                    if (croppedByteString != null) {
                        serverComm.sendSupplier(() -> InputFrame.newBuilder()
                                .setPayloadType(PayloadType.IMAGE)
                                .addPayloads(croppedByteString)
                                .build(), SOURCE, true);
                    }
                } else {
                    ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
                    image.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
                    ByteString jpegByteString = ByteString.copyFrom(byteArrayOutputStream.toByteArray());
                    serverComm.sendSupplier(() -> InputFrame.newBuilder()
                            .setPayloadType(PayloadType.IMAGE)
                            .addPayloads(jpegByteString)
                            .build(), SOURCE, true);
                }
            }
        }

        logList.add(TAG + ": Stop: " + SystemClock.uptimeMillis() + "\n");
    }
}
