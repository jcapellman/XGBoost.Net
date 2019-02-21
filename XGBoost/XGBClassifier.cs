using System;
using System.Collections.Generic;
using System.Linq;

namespace XGBoost
{
    public class XGBClassifier : XGBModelBase
    {
        public XGBClassifier(int maxDepth = 3,
                             float learningRate = 0.1F,
                             int nEstimators = 100,
                             bool silent = true,
                             string objective = "binary:logistic",
                             int nJobs = -1,
                             float gamma = 0,
                             int minChildWeight = 1,
                             int maxDeltaStep = 0,
                             float subsample = 1,
                             float colSampleByTree = 1,
                             float colSampleByLevel = 1,
                             float regAlpha = 0,
                             float regLambda = 1,
                             float scalePosWeight = 1,
                             float baseScore = 0.5F,
                             int seed = 0,
                             float missing = float.NaN)
        {
            parameters["max_depth"] = maxDepth;
            parameters["learning_rate"] = learningRate;
            parameters["n_estimators"] = nEstimators;
            parameters["silent"] = silent;
            parameters["objective"] = objective;
            parameters["n_jobs"] = nJobs;
            parameters["gamma"] = gamma;
            parameters["min_child_weight"] = minChildWeight;
            parameters["max_delta_step"] = maxDeltaStep;
            parameters["subsample"] = subsample;
            parameters["colsample_bytree"] = colSampleByTree;
            parameters["colsample_bylevel"] = colSampleByLevel;
            parameters["reg_alpha"] = regAlpha;
            parameters["reg_lambda"] = regLambda;
            parameters["scale_pos_weight"] = scalePosWeight;
            parameters["base_score"] = baseScore;
            parameters["seed"] = seed;
            parameters["missing"] = missing;
            parameters["_Booster"] = null;
        }


        public XGBClassifier(IDictionary<string, object> newParams)
        {
            parameters = newParams;
        }

        public void Fit(float[][] data, float[] labels)
        {
            using (var train = new DMatrix(data, labels))
            {
                booster = Train(parameters, train, ((int)parameters["n_estimators"]));
            }
        }

        public static Dictionary<string, object> GetDefaultParameters()
        {
            var defaultParameters = new Dictionary<string, object>
            {
                ["max_depth"] = 3,
                ["learning_rate"] = 0.1f,
                ["n_estimators"] = 100,
                ["silent"] = true,
                ["objective"] = "binary:logistic",
                ["n_jobs"] = -1,
                ["gamma"] = 0,
                ["min_child_weight"] = 1,
                ["max_delta_step"] = 0,
                ["subsample"] = 1,
                ["colsample_bytree"] = 1,
                ["colsample_bylevel"] = 1,
                ["reg_alpha"] = 0,
                ["reg_lambda"] = 1,
                ["scale_pos_weight"] = 1,
                ["base_score"] = 0.5f,
                ["seed"] = 0,
                ["missing"] = float.NaN,
                ["_Booster"] = null
            };

            return defaultParameters;
        }

        public void SetParameter(string parameterName, object parameterValue)
        {
            parameters[parameterName] = parameterValue;
        }

        private static float SigmoidSum(IEnumerable<float> vector)
        {
            return 1.0f / (1.0f + (float)System.Math.Exp(-vector.Sum()));
        }

        public (float confidence, float[] featureContribs) Predict(float[][] data)
        {
            using (var test = new DMatrix(data))
            {
                // Get the individual feature contributions to the prediction
                IEnumerable<float> featurePredictions = booster.Predict(test, predContribs: true);
                // Calculate the overall prediction confidence and concatenate it with the feature contributions
                float predictionConfidence = booster.Predict(test)[0];
                return (predictionConfidence, featurePredictions.ToArray());
            }
        }

        private static XGBooster Train(IDictionary<string, object> parameters, DMatrix dTrain, int numBoostRound = 10)
        {
            XGBooster booster = new XGBooster(parameters, dTrain);
            for (int round = 0; round < numBoostRound; round++)
            {
                booster.Update(dTrain, round);
            }
            return booster;
        }
    }
}
