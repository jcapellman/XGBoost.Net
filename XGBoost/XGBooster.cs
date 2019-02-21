using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.InteropServices;

using XGBoost.lib;

namespace XGBoost
{
    public class XGBooster : IDisposable
    {
        private bool disposed;
        private readonly IntPtr handle;
        private const int NormalOptionMask = 0;
        private const int ContribsOptionMask = 4;
        public IntPtr Handle => handle;

        public XGBooster(IDictionary<string, object> parameters, DMatrix train)
        {
            var dmats = new[] { train.Handle };
            var len = unchecked((ulong)dmats.Length);
            var output = XGBoostNative.XGBoosterCreate(dmats, len, out handle);
            if (output == -1)
            {
                throw new DllFailException(XGBoostNative.XGBGetLastError());
            }
            SetParameters(parameters);
        }

        public XGBooster()
        {
            IntPtr tempPtr;
            var output = XGBoostNative.XGBoosterCreate(null, 0, out tempPtr);
            if(output == -1)
            {
                throw new DllFailException(XGBoostNative.XGBGetLastError());
            }
            handle = tempPtr;
        }

        public XGBooster(DMatrix train)
        {
            var dmats = new[] { train.Handle };
            var len = unchecked((ulong)dmats.Length);
            var output = XGBoostNative.XGBoosterCreate(dmats, len, out handle);
            if (output == -1) 
            {
                throw new DllFailException(XGBoostNative.XGBGetLastError());
            }
        }

        public XGBooster(string fileName, int silent = 1)
        {
            IntPtr tempPtr;
            var newBooster = XGBoostNative.XGBoosterCreate(null, 0, out tempPtr);
            var output = XGBoostNative.XGBoosterLoadModel(tempPtr, fileName);
            if (output == -1)
            {
                throw new DllFailException(XGBoostNative.XGBGetLastError());
            }
            handle = tempPtr;
        }

        public void Update(DMatrix train, int iter)
        {
            var output = XGBoostNative.XGBoosterUpdateOneIter(Handle, iter, train.Handle);
            if (output == -1)
            {
                throw new DllFailException(XGBoostNative.XGBGetLastError());
            }
        }

        public float[] Predict(DMatrix test, bool predContribs = false, int ntreeLimit = 0)
        {
            ulong predsLen;
            IntPtr predsPtr;
            int optionsMask = predContribs ? ContribsOptionMask : NormalOptionMask;
            var output = XGBoostNative.XGBoosterPredict(
                handle, test.Handle, optionsMask, ntreeLimit, out predsLen, out predsPtr);
            if (output == -1)
            {
                throw new DllFailException(XGBoostNative.XGBGetLastError());
            }
            return GetPredictionsArray(predsPtr, predsLen);
        }

        public float[] GetPredictionsArray(IntPtr predsPtr, ulong predsLen)
        {
            var length = unchecked((int)predsLen);
            var preds = new float[length];
            for (var i = 0; i < length; i++)
            {
                var floatBytes = new byte[4];
                for (var b = 0; b < 4; b++)
                {
                    floatBytes[b] = Marshal.ReadByte(predsPtr, 4 * i + b);
                }
                preds[i] = BitConverter.ToSingle(floatBytes, 0);
            }
            return preds;
        }

        public void SetParameters(IDictionary<string, object> parameters)
        {
            // support internationalisation i.e. support floats with commas (e.g. 0,5F)
            var nfi = new NumberFormatInfo { NumberDecimalSeparator = "." };

            SetParameter("max_depth", ((int)parameters["max_depth"]).ToString());
            SetParameter("learning_rate", ((float)parameters["learning_rate"]).ToString(nfi));
            SetParameter("n_estimators", ((int)parameters["n_estimators"]).ToString());
            SetParameter("silent", ((bool)parameters["silent"]).ToString());
            SetParameter("objective", (string)parameters["objective"]);

            SetParameter("n_jobs", ((int)parameters["n_jobs"]).ToString());
            SetParameter("gamma", ((float)parameters["gamma"]).ToString(nfi));
            SetParameter("min_child_weight", ((int)parameters["min_child_weight"]).ToString());
            SetParameter("max_delta_step", ((int)parameters["max_delta_step"]).ToString());
            SetParameter("subsample", ((float)parameters["subsample"]).ToString(nfi));
            SetParameter("colsample_bytree", ((float)parameters["colsample_bytree"]).ToString(nfi));
            SetParameter("colsample_bylevel", ((float)parameters["colsample_bylevel"]).ToString(nfi));
            SetParameter("reg_alpha", ((float)parameters["reg_alpha"]).ToString(nfi));
            SetParameter("reg_lambda", ((float)parameters["reg_lambda"]).ToString(nfi));
            SetParameter("scale_pos_weight", ((float)parameters["scale_pos_weight"]).ToString(nfi));

            SetParameter("base_score", ((float)parameters["base_score"]).ToString(nfi));
            SetParameter("seed", ((int)parameters["seed"]).ToString());
            SetParameter("missing", ((float)parameters["missing"]).ToString(nfi));
        }

        public void SetParametersGeneric(IDictionary<string, object> parameters)
        {
            foreach (var param in parameters)
            {
                if (param.Value != null)
                {
                    SetParameter(param.Key, param.Value.ToString());
                }
            }
        }

        public void PrintParameters(IDictionary<string, object> parameters)
        {
            Console.WriteLine("max_depth: " + (int)parameters["max_depth"]);
            Console.WriteLine("learning_rate: " + (float)parameters["learning_rate"]);
            Console.WriteLine("n_estimators: " + (int)parameters["n_estimators"]);
            Console.WriteLine("silent: " + (bool)parameters["silent"]);
            Console.WriteLine("objective: " + (string)parameters["objective"]);

            Console.WriteLine("n_jobs: " + (int)parameters["n_jobs"]);
            Console.WriteLine("gamma: " + (float)parameters["gamma"]);
            Console.WriteLine("min_child_weight: " + (int)parameters["min_child_weight"]);
            Console.WriteLine("max_delta_step: " + (int)parameters["max_delta_step"]);
            Console.WriteLine("subsample: " + (float)parameters["subsample"]);
            Console.WriteLine("colsample_bytree: " + (float)parameters["colsample_bytree"]);
            Console.WriteLine("colsample_bylevel: " + (float)parameters["colsample_bylevel"]);
            Console.WriteLine("reg_alpha: " + (float)parameters["reg_alpha"]);
            Console.WriteLine("reg_lambda: " + (float)parameters["reg_lambda"]);
            Console.WriteLine("scale_pos_weight: " + (float)parameters["scale_pos_weight"]);

            Console.WriteLine("base_score: " + (float)parameters["base_score"]);
            Console.WriteLine("seed: " + (int)parameters["seed"]);
            Console.WriteLine("missing: " + (float)parameters["missing"]);
        }

        public void SetParameter(string name, string val)
        {
            int output = XGBoostNative.XGBoosterSetParam(handle, name, val);
            if (output == -1)
            {
                throw new DllFailException(XGBoostNative.XGBGetLastError());
            }
        }

        public void Save(string fileName)
        {
            XGBoostNative.XGBoosterSaveModel(handle, fileName);
        }

        public string[] DumpModelEx(string fmap, int with_stats, string format)
        {
            int length;
            string[] dumpStr;
            XGBoostNative.XGBoosterDumpModel(handle, fmap, with_stats, out length, out dumpStr);
            return dumpStr;
        }

        public byte[] GetModelRaw()
        {
            int length;
            IntPtr dumpPtr;
            int output = XGBoostNative.XGBoosterGetModelRaw(handle, out length, out dumpPtr);
            if (output == -1)
            {
                throw new DllFailException(XGBoostNative.XGBGetLastError());
            }
            byte[] modelBytes = new byte[length];
            Marshal.Copy(dumpPtr, modelBytes, 0, length);
            return modelBytes;
        }

        public void LoadModelFromBuffer(byte[] buf)
        {
            int output = XGBoostNative.XGBoosterLoadModelFromBuffer(handle, buf, buf.Length);
            if (output == -1)
            {
                throw new DllFailException(XGBoostNative.XGBGetLastError());
            }
        }

        // Dispose pattern from MSDN documentation
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposed) return;
            XGBoostNative.XGDMatrixFree(handle);
            disposed = true;
        }
    }
}
