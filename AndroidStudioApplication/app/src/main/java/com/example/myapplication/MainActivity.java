package com.example.myapplication;

import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import android.util.Log;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;

//import androidx.activity.EdgeToEdge;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;





public class MainActivity extends AppCompatActivity {

    private static final String TAG = MainActivity.class.getName();

    private EditText myEditText;
    private TextView myTextView;
    private Button myButton;

    private Translator myTranslator;

    private Handler myHandler = new Handler(Looper.getMainLooper()){
        @Override
        public void handleMessage(@NonNull Message msg){
            super.handleMessage(msg);
            String result = (String)msg.obj;
            showTranslationResult(result);
            myButton.setEnabled(true);
        }
    };

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        myButton = findViewById(R.id.TranslateBtn1);
        myEditText = findViewById(R.id.editForm1);
        myTextView = findViewById(R.id.viewForm1);

        myButton.setOnClickListener(v -> {

            myButton.setEnabled(false);
            String textToTranslate = myEditText.getText().toString();
            Thread thread = new Thread(() -> {
                Looper.prepare();
                final String result;
                try {
                    result = translate(textToTranslate);
                } catch (InterruptedException | JSONException e) {
                    throw new RuntimeException(e);
                }
                Message msg = Message.obtain();
                msg.obj = result;
                myHandler.sendMessage(msg);
            });
            thread.start();
        });

    }
    private void showTranslationResult(String result) {
        myTextView.setText(result);
    }


    public static String getText(ArrayList<Long> result_idx, JSONObject idx2word)
    {
        StringBuilder outputs = new StringBuilder();
        try {
            for(int i = 0;i < result_idx.size(); i++)
            {
                String targetWord = idx2word.getString(""+result_idx.get(i));
                if(targetWord.startsWith("▁"))
                {
                    targetWord = targetWord.replace("▁"," ");
                }
                outputs.append(targetWord);
            }
        }
        catch (JSONException e) {
            Log.e(TAG, "JSONException ", e);
        }
        return outputs.toString();
    }

    private String translate(final String text) throws InterruptedException, JSONException {

        if(myTranslator==null)
        {
            try{
                myTranslator = new Translator(getApplicationContext(), "model.onnx");
            }
            catch(IOException e){
                Log.e(TAG, "Error reading assets", e);
                finish();
            }
        }

        JSONObject word2idx;
        JSONObject idx2word;
        JSONloader jl = new JSONloader();
        word2idx = jl.jsonloader(getApplicationContext(),"word2idx.json");
        idx2word = jl.jsonloader(getApplicationContext(),"idx2word.json");

        ArrayList<Long> inputs;
        inputs = Tokenizer.tokenize(text, word2idx);
        Map<Integer, String>Mark_idx = new HashMap<>();
        String tgt;
        int inputs_length = Objects.requireNonNull(inputs).size();
        Log.d(TAG,""+inputs_length);
        for(int i=0;i<inputs.size();i++)
            Log.d(TAG,""+inputs.get(i));
        int idx = 0;
        for(int i = 0;i < inputs.size();i++ ,idx++)
        {
            tgt = ""+idx2word.get(""+inputs.get(i));
            Log.d(TAG,"tst is "+tgt);
            if(tgt.equals("\"") || tgt.equals("'") || tgt.equals(".") || tgt.equals(","))
            {
                Mark_idx.put(idx,tgt);
                Log.d(TAG,""+idx);
                inputs.remove(i);
                i--;
            }
        }
        for(int i=0;i<inputs.size();i++)
            Log.d(TAG,""+inputs.get(i));

        Log.d(TAG,"mask is "+ "0 :" +Mark_idx.get(0));
        Log.d(TAG,"mask is "+ "3 :" +Mark_idx.get(3));


        ArrayList<Long> result_idx = new ArrayList<>();
        FloatBuffer probBuffer;

        for (int i = 0; i < inputs_length - 1; i++) {


            if(Mark_idx.containsKey(i))
            {
                result_idx.add(word2idx.getLong(Mark_idx.get(i)));
                continue;
            }
            Log.d(TAG,"inputs is "+inputs.get(0));
            probBuffer = myTranslator.runModule(inputs, inputs.size(), config.BOS_TOKEN);
            long result = myTranslator.generate(probBuffer);
            Log.d(TAG,"output is "+result);
            result_idx.add(result);
            inputs.remove(0);
        }
        return getText(result_idx, idx2word);

    }

}