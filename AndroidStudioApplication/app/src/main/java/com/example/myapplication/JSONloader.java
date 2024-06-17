package com.example.myapplication;

import android.content.Context;
import android.util.Log;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;


public class JSONloader {

    private static final String TAG = JSONloader.class.getName();
    public JSONObject jsonloader(Context context ,String Filename)
    {
        String json;
        JSONObject jsonobject;
        try {
            InputStream is = context.getAssets().open(Filename);
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();

            json = new String(buffer, StandardCharsets.UTF_8);
            jsonobject = new JSONObject(json);
        } catch (JSONException | IOException e) {
            Log.e(TAG, "JSONException | IOException ", e);
            return null;
        }
        return jsonobject;
    }
}
