using System.Collections.Generic;

namespace XGBoost
{
    public class XGBRegressor : XGBModelBase
    {
        public XGBRegressor(int maxDepth = 3,
                            float learningRate = 0.1F,
                            int nEstimators = 100,
                            bool silent = true,
                            string objective = "reg:linear",
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

        public void Fit(float[][] data, float[] labels)
        {
            using (var train = new DMatrix(data, labels))
            {
                booster = Train(parameters, train, (int)parameters["n_estimators"]);
            }
        }

        public float[] Predict(float[][] data)
        {
            using (var test = new DMatrix(data))
            {
                return booster.Predict(test);
            }
        }

        private XGBooster Train(IDictionary<string, object> args, DMatrix train, int numBoostRound = 10)
        {
            var bst = new XGBooster(args, train);
            for (var i = 0; i < numBoostRound; i++) { bst.Update(train, i); }
            return bst;
        }
    }
}
