package com.example.tfltest;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.core.BaseOptions;
import org.tensorflow.lite.task.vision.classifier.Classifications;
import org.tensorflow.lite.task.vision.classifier.ImageClassifier;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;
import org.tensorflow.lite.gpu.CompatibilityList;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedDeque;

public class Pipeline {
    private static final String TAG = "Pipeline";
    ImageClassifier imageClassifier = null;
    ObjectDetector objectDetector = null;
    ConcurrentLinkedDeque<String> logList;
    final long FRAME_INPUT_INTERVAL_MS;

    public Pipeline(MainActivity mainActivity) {
        this.logList = mainActivity.logList;
        this.FRAME_INPUT_INTERVAL_MS = mainActivity.FRAME_INPUT_INTERVAL_MS;
        try {
            ImageClassifier.ImageClassifierOptions.Builder optionsBuilder =
                    ImageClassifier.ImageClassifierOptions.builder()
                            .setScoreThreshold(0.4f)
                            .setMaxResults(1);

            BaseOptions.Builder baseOptionsBuilder = BaseOptions.builder().setNumThreads(1);

            // Works on Google Glass even though
            // (new CompatibilityList()).isDelegateSupportedOnThisDevice() returns false.
            baseOptionsBuilder.useGpu();

            optionsBuilder.setBaseOptions(baseOptionsBuilder.build());
            File modelFile = new File("/sdcard/tflite_models/stirling_r50.tflite");

            imageClassifier = ImageClassifier.createFromFileAndOptions(
                    modelFile, optionsBuilder.build());
        } catch (IOException e) {
            e.printStackTrace();
        }

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

    private boolean classifyImage(Bitmap image, String correctClass) {
        ImageProcessor imageProcessor = (new ImageProcessor.Builder()).build();
        TensorImage tensorImage = imageProcessor.process(TensorImage.fromBitmap(image));

        List<Detection> detections = objectDetector.detect(tensorImage);
        if (detections.size() == 0) {
            return false;
        }
        if (detections.size() > 1) {
            throw new RuntimeException();
        }
        RectF rectF = detections.get(0).getBoundingBox();

        if ((rectF.bottom < 0) || (rectF.top < 0) || (rectF.width() == 0) ||
                (rectF.height() == 0)) {
            return false;
        }

        image = Bitmap.createBitmap(image, (int)rectF.left, (int)rectF.top,
                (int)rectF.width(), (int)rectF.height());

        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        image.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
        ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(
                byteArrayOutputStream.toByteArray());
        image = BitmapFactory.decodeStream(byteArrayInputStream);

        imageProcessor = (new ImageProcessor.Builder()).build();
        tensorImage = imageProcessor.process(TensorImage.fromBitmap(image));

        List<Classifications> results = imageClassifier.classify(tensorImage);
        if ((results.size() != 1) || (results.get(0).getCategories().size() == 0)) {
            return false;
        }

        return results.get(0).getCategories().get(0).getLabel().equals(correctClass);
    }

    public void testPipeline() {
        int good = 0;
        int bad = 0;

        classifyImage(BitmapFactory.decodeFile("/sdcard/test_images/stirling/1screw/2_frame-0000.jpg"), "warmup");
        Log.d(TAG, "Warmup finished.");

        File testImages = new File( Environment.getExternalStorageDirectory().getPath()
                + "/test_images/stirling");
        final int folderCount = testImages.list().length;
        final int testImagePerFolder = 150;
        final int imageCount = testImagePerFolder * folderCount;
        File[] classDirs = testImages.listFiles();

        logList.add(TAG + ": Total Images: " + imageCount + "\n");
        logList.add(TAG + ": Start: " + SystemClock.uptimeMillis() + "\n");

        long last_frame_time = SystemClock.uptimeMillis() - FRAME_INPUT_INTERVAL_MS;
        for (int i = 0; i < folderCount; i++) {
            File classDir = classDirs[i];
            String correctClass = classDir.getName();
//            final int imageCount = classDir.list().length;
//            Log.d(TAG, "Total Images: " + imageCount);

            File[] imageFiles = classDir.listFiles();
            for (int j = 0; j < testImagePerFolder; j++) {
                File imageFile = imageFiles[j];
                Bitmap image = BitmapFactory.decodeFile(imageFile.getPath());
                long cur_frame_time = SystemClock.uptimeMillis();
                if (last_frame_time + FRAME_INPUT_INTERVAL_MS > cur_frame_time) {
                    try {
                        Thread.sleep(Math.max(last_frame_time + FRAME_INPUT_INTERVAL_MS - cur_frame_time, 0));
                    } catch (InterruptedException e) {
                        Log.w(TAG, "Thread interrupted.");
                    }
                }
                if (classifyImage(image, correctClass)) {
                    good++;
                } else {
                    bad++;
                }
                last_frame_time += FRAME_INPUT_INTERVAL_MS;
            }
        }

        logList.add(TAG + ": Stop: " + SystemClock.uptimeMillis() + "\n");
    }

    public void testPHashPipeline() {
        int good = 0;
        int bad = 0;

        classifyImage(BitmapFactory.decodeFile("/sdcard/test_images/stirling/1screw/2_frame-0000.jpg"), "warmup");
        Log.d(TAG, "Warmup finished.");

        File testImages = new File( Environment.getExternalStorageDirectory().getPath()
                + "/test_images/stirling");
        final int folderCount = testImages.list().length;
        final int testImagePerFolder = 150;
        final int imageCount = testImagePerFolder * folderCount;
        File[] classDirs = testImages.listFiles();
        long lastPHash = 0;
        int uniqueCount = 0;
        final int DIFF_THRESHOLD = 2;

        logList.add(TAG + ": Total Images: " + imageCount + "\n");
        logList.add(TAG + ": Start: " + SystemClock.uptimeMillis() + "\n");

        long last_frame_time = SystemClock.uptimeMillis() - FRAME_INPUT_INTERVAL_MS;
        for (int i = 0; i < folderCount; i++) {
            File classDir = classDirs[i];
            String correctClass = classDir.getName();
//            final int imageCount = classDir.list().length;
//            Log.d(TAG, "Total Images: " + imageCount);

            File[] imageFiles = classDir.listFiles();
            Arrays.sort(imageFiles);

            for (int j = 0; j < testImagePerFolder; j++) {
                File imageFile = imageFiles[j];
                Bitmap image = BitmapFactory.decodeFile(imageFile.getPath());

                long cur_frame_time = SystemClock.uptimeMillis();
                if (last_frame_time + FRAME_INPUT_INTERVAL_MS > cur_frame_time) {
                    try {
                        Thread.sleep(Math.max(last_frame_time + FRAME_INPUT_INTERVAL_MS - cur_frame_time, 0));
                    } catch (InterruptedException e) {
                        Log.w(TAG, "Thread interrupted.");
                    }
                }
                long curPHash = ImagePHash.pHash(image);
                if (ImagePHash.distance(lastPHash, curPHash) >= DIFF_THRESHOLD) {
                    uniqueCount++;
                    lastPHash = curPHash;
                    if (classifyImage(image, correctClass)) {
                        good++;
                    } else {
                        bad++;
                    }
                }
                last_frame_time += FRAME_INPUT_INTERVAL_MS;
            }
        }
//        Log.d(TAG, "Unique Images: " + uniqueCount);

        logList.add(TAG + ": Unique Images: " + uniqueCount + "\n");
        logList.add(TAG + ": Stop: " + SystemClock.uptimeMillis() + "\n");
    }
}
