using System;
using System.Collections.Generic;

namespace XGBoost
{
    public class XGBModelBase : IDisposable
    {
        protected IDictionary<string, object> parameters = new Dictionary<string, object>();
        protected XGBooster booster;

        public void SaveModelToFile(string fileName)
        {
            booster.Save(fileName);
        }

        public static XGBClassifier LoadClassifierFromFile(string fileName)
        {
            return new XGBClassifier { booster = new XGBooster(fileName) };
        }

        public static XGBRegressor LoadRegressorFromFile(string fileName)
        {
            return new XGBRegressor { booster = new XGBooster(fileName) };
        }

        public string[] DumpModelEx(string fmap = "", int with_stats = 0, string format = "json")
        {
            return booster.DumpModelEx(fmap, with_stats, format);
        }

        public void Dispose()
        {
            if (booster != null)
            {
                booster.Dispose();
            }
        }

        public XGBClassifier Clone()
        {
            byte[] modelBytes = booster.GetModelRaw();
            XGBooster newBooster = new XGBooster();
            newBooster.LoadModelFromBuffer(modelBytes);

            return new XGBClassifier
            {
                booster = newBooster,
                parameters = new Dictionary<string, object>(parameters)
            };
        }
    }
}
