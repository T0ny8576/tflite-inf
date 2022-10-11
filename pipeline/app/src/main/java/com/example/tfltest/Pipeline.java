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
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class Pipeline {
    private static final String TAG = "Pipeline";
    ImageClassifier imageClassifier = null;
    ObjectDetector objectDetector = null;
    FileWriter logFileWriter;

    public Pipeline(MainActivity mainActivity) {
        this.logFileWriter = mainActivity.logFileWriter;
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

            File modelFile = new File("/sdcard/tflite_models/sitrling_all_classes.tflite");
            objectDetector = ObjectDetector.createFromFileAndOptions(
                    modelFile, optionsBuilder.build());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void runTest() {
        int good = 0;
        int bad = 0;

        classifyImage("/sdcard/test_images/stirling/1screw/2_frame-0000.jpg", "warmup");
        Log.d(TAG, "Warmup finished.");

        File testImages = new File( Environment.getExternalStorageDirectory().getPath()
                + "/test_images/stirling");
        for (int i = 0; i < 1; i++) {
            File classDir = testImages.listFiles()[i];
            String correctClass = classDir.getName();
//            final int imageCount = classDir.list().length;
//            Log.d(TAG, "Total Images: " + imageCount);
            final int imageCount = 400;
            try {
                logFileWriter.write(TAG + ": Total Images: " + imageCount + "\n");
            } catch (IOException e) {
                e.printStackTrace();
            }
            try {
                logFileWriter.write(TAG + ": Start: " + SystemClock.uptimeMillis() + "\n");
            } catch (IOException e) {
                e.printStackTrace();
            }
            for (int j = 0; j < 400; j++) {
                File imageFile = classDir.listFiles()[j];
                if (classifyImage(imageFile.getPath(), correctClass)) {
                    good++;
                } else {
                    bad++;
                }

            }
            try {
                logFileWriter.write(TAG + ": Stop: " + SystemClock.uptimeMillis() + "\n");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private boolean classifyImage(String imagePath, String correctClass) {
        Bitmap image = BitmapFactory.decodeFile(imagePath);
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
}
