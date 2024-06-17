package com.example.myapplication;

import android.content.Context;
import android.util.Log;

import java.io.IOException;

import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;


import ai.onnxruntime.OnnxTensor;
//import ai.onnxruntime.OnnxTensorLike;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class Translator {

    private final OrtEnvironment environment;

    private OrtSession ortsession;

    private static final String TAG = Translator.class.getName();

    Translator(Context context, String moduleAssetName) throws IOException
    {

        //创建onnx runtime运行上下文环境
        environment = OrtEnvironment.getEnvironment();

        //创建onnx runtime会话选项
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();



        try {


            //读入模型
            ortsession = environment.createSession(utils.assetFilePath(context.getApplicationContext(), moduleAssetName), options);

            Log.d(TAG, "Loading module succeed");

        } catch (OrtException e) {
            //ort环境会话异常
            Log.e(TAG, "ortSession create Error");
        } catch (IOException e)
        {
            Log.e(TAG,"IOexception");
        }
    }

    public FloatBuffer runModule(ArrayList<Long> inputs, int length, long target_idx)
    {
        long[] inputTensorShape = new long[]{1, length};
        long[] attn_maskTensorShape = new long[]{1, length};
        long[] dec_input_idsTensorShape = new long[]{1, 1};

        long[] input_idxs = new long[length];
        long[] attn_mask = new long[length];
        long[] dec_input_idxs = new long[1];

        //将输入的idx序列存入数组
        for(int i = 0;i < length;i++)
        {
            input_idxs[i] = inputs.get(i);
        }
        //将attn_mask数组全都赋值为1
        Arrays.fill(attn_mask,1);
        //将上一次得到的目标idx传入张量
        dec_input_idxs[0] = target_idx;

        //创建两个输入变量 shape都是[batch_size,seq_len]
        LongBuffer inputTensorBuffer = LongBuffer.wrap(input_idxs);
        LongBuffer attn_maskTensorBuffer = LongBuffer.wrap(attn_mask);
        LongBuffer dec_input_idsTensorBuffer = LongBuffer.wrap(dec_input_idxs);
        OnnxTensor inputTensor;
        OnnxTensor attn_maskTensor;
        OnnxTensor dec_input_idsTensor;
        try {

            inputTensor = OnnxTensor.createTensor(
                    environment,inputTensorBuffer,inputTensorShape);
            attn_maskTensor = OnnxTensor.createTensor(
                    environment, attn_maskTensorBuffer, attn_maskTensorShape);
            dec_input_idsTensor = OnnxTensor.createTensor(
                    environment, dec_input_idsTensorBuffer, dec_input_idsTensorShape);

        }
        catch (OrtException e)
        {
            Log.e(TAG,"Tensor create failed");
            return null;
        }

        //构建传入模型的map映射
        Map<String,OnnxTensor>inputMap = new HashMap<>();
        inputMap.put("input_ids",inputTensor);
        inputMap.put("attention_mask",attn_maskTensor);
        inputMap.put("onnx::Reshape_2",dec_input_idsTensor);
        OrtSession.Result ort_result;
        //运行模型
        try {
            ort_result = ortsession.run(inputMap);
        } catch (OrtException e) {
            Log.e(TAG,"model run failed");
            return null;
        }

        //从返回值中提取目标概率分布
        float[] result;
        try {
            result = ((float[][][])ort_result.get(0).getValue())[0][0];
        } catch (OrtException e) {
            Log.e(TAG,"get value of result failed");
            return null;
        }
        Log.d(TAG,"RATE:"+Arrays.toString(result));
        //将结果概率分布序列转换成浮点流输出
        return FloatBuffer.wrap(result);
    }

    public int generate(FloatBuffer resultBuffer) {
        float[] probs = resultBuffer.array();
        float max_prob = probs[0];
        int idx = 0;

        for (int i = 1; i < probs.length; i++)
        {
            if(probs[i] > max_prob)
            {
                max_prob = probs[i];
                idx = i;
            }
        }
        return idx;
    }

}




