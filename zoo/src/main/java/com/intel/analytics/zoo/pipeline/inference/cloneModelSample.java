package com.intel.analytics.zoo.pipeline.inference;

import java.util.*;

public class cloneModelSample extends AbstractInferenceModel {
    public cloneModelSample(int supportedConcurrentNum){
        super(supportedConcurrentNum);
    }

    public List<List<JTensor>> doTextClassificationPredict(){
        // text classification predict
        System.out.println("Begin to load embedding and do pre-processing.\n");
        int stopWordsCount = 1;
        int sequenceLength = 500;
        String sampleText = "it is for test";
        TextPreprocessor preprocessor = new TextPreprocessor();
        String embeddingPath = System.getProperty("EMBEDDING_PATH");
        Map<String, List<Float>> embMap = preprocessor.loadEmbedding(embeddingPath);
        JTensor input = preprocessor.preprocessWithEmbMap(sampleText, stopWordsCount, sequenceLength, embMap);
        System.out.println("Pre-processing has finished.\n");
        List<JTensor> inputList = new ArrayList<JTensor>();
        inputList.add(input);
        long begin = System.currentTimeMillis();
        List<List<JTensor>> result = predict(inputList);
        long end = System.currentTimeMillis();
        long predictTime = end - begin;
        System.out.println("Predict Time elapsed: " + predictTime);
        return result;
    }

    public List<List<JTensor>> doAutogradPredict(){
        // autograd predict
        List<Float> randomList = new ArrayList<Float>();
        for(int i = 0; i < 2000; i++){
            randomList.add((float)Math.random());
        }
        List<Integer> shape = new ArrayList<Integer>();
        shape.add(1000);
        shape.add(2);
        JTensor input = new JTensor(randomList, shape);

        List<JTensor> inputList = new ArrayList<JTensor>();
        inputList.add(input);
        long begin = System.currentTimeMillis();
        List<List<JTensor>> result = predict(inputList);
        long end = System.currentTimeMillis();
        long predictTime = end - begin;
        System.out.println("Predict Time elapsed: " + predictTime);
        return result;
    }

    // Unit Test for loading models sharing weights, the predict part is optional.
    public static void main(String[] args) {
        String baseDir = System.getProperty("baseDir", "/home/yidiyang/workspace");
        String modelPath = System.getProperty("modelPath", baseDir + "/model/autogradModel");
        int num = 100;
        long begin = System.currentTimeMillis();
        cloneModelSample model = new cloneModelSample(num);
        model.load(modelPath);
        long end = System.currentTimeMillis();
        long loadTime = end - begin;
        System.out.println("Total "+num+" models sharing weights have been loaded. Time elapsed:" + loadTime + ".\n");

        //do predicting according to the model and input
//        List<List<JTensor>> result = model.doAutogradPredict();
//
//        JTensor resultTensor = result.get(0).get(0);
//        float[] resultDis = resultTensor.getData();
//        int resultClass = 0;
//        float maxProb = 0;
//        for (int i = 0; i < resultDis.length; i++) {
//            if (resultDis[i] >= maxProb) {
//                resultClass = i;
//                maxProb = resultDis[i];
//            }
//        }
    }
}
