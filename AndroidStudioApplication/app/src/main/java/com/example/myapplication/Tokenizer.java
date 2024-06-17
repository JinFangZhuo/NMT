package com.example.myapplication;

import android.util.Log;

import org.json.JSONException;
import org.json.JSONObject;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.regex.*;

public class Tokenizer
{
    private static final String TAG = Tokenizer.class.getName();

    public static int getTextLength(String text)
    {
        return text.split(" ").length;
    }
    public static ArrayList<Long> tokenize(String text , JSONObject word2idx)
    {

        Pattern pattern = Pattern.compile("[a-zA-Z]+|\\p{Punct}|\\s+");
        Matcher matcher = pattern.matcher(text);

        ArrayList<String> inputWord = new ArrayList<>();

        boolean exist_space = false;
        boolean first_word = true;

        while (matcher.find()) {
            String match = matcher.group();
            if(match.equals("."))
            {
                first_word = true;
            }
            if (match.matches("\\s+")) {
                exist_space = true;
                continue; // Skip the spaces themselves
            }
            if (exist_space) {
                match = "▁" + match;
                exist_space = false;
            }if(match.matches("[a-zA-Z]+") && first_word)
            {
                inputWord.add("▁"+match);
                first_word = false;
                continue;
            }
            inputWord.add(match);
        }
        for(int i=0;i<inputWord.size();i++)
            Log.d(TAG,""+inputWord.get(i));
        ArrayList<Long> inputs = new ArrayList<Long>();
        try {
            for (int i = 0; i < inputWord.size(); i++) {
                while(!inputWord.get(i).isEmpty())
                {
                    int len = config.MAX_WORD_LENGTH;
                    if(inputWord.get(i).length() < len)
                    {
                        len = inputWord.get(i).length();//如果目标字符串比最大长度要小，则匹配字符串变为目标字符串长度
                    }
                    String TryWord = inputWord.get(i).substring(0, len);
                    while(!word2idx.has(TryWord))//找不到匹配字符串则继续循环
                    {
                        if(inputWord.get(i).length()==1)//如果目标字符串为长度一，则直接返回
                            break;
                        TryWord = TryWord.substring(0,TryWord.length()-1);
                    }
                    inputWord.set(i,inputWord.get(i).substring(TryWord.length()));
                    if(word2idx.has(TryWord))
                        inputs.add(word2idx.getLong(TryWord));
                    else
                        inputs.add(word2idx.getLong("<unk>"));
                }
            }
            inputs.add(word2idx.getLong("</s>"));
        }
        catch (
                JSONException e) {
            android.util.Log.e(TAG, "JSONException ", e);
            return null;
        }
        return inputs;
    }

}
