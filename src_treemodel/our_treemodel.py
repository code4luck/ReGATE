# 用于实现自己的unikp模型-----极限树模型
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_pcc(y_true, y_pred):
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def safe_scc(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "pcc": safe_pcc(y_true, y_pred),
        "scc": safe_scc(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mse": float(mse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def load_inputs_pkl(train_path, test_path):
    # 用于加载 输入的表征一行行的记录
    train_features = np.load(train_path + "_features.npy")
    train_labels = np.load(train_path + "_labels.npy")
    test_features = np.load(test_path + "_features.npy")
    test_labels = np.load(test_path + "_labels.npy")

    # 输出基本信息
    print(f"Train features shape: {train_features.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test features shape: {test_features.shape}")
    print(f"Test labels shape: {test_labels.shape}")

    return train_features, train_labels, test_features, test_labels


def train_eval_one_seed(
    x_train,
    y_train,
    x_test,
    y_test,
    output_dir,
    seed,
    n_estimators,
    n_jobs,
):
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=n_jobs,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = compute_metrics(y_test, y_pred)
    
    # 创建一个pred df 进行保存预测结果
    pred_df = pd.DataFrame()
    pred_df["label"] = y_test
    pred_df["prediction"] = y_pred
    pred_df["seed"] = seed

    pred_path = output_dir / f"test_predictions_seed_{seed}.csv"
    pred_df.to_csv(pred_path, index=False)

    model_path = output_dir / f"extratrees_seed_{seed}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    metrics_row = {
        "seed": seed,
        "train_size": len(y_train),
        "test_size": len(y_test),
        **metrics,
        "prediction_path": str(pred_path),
        "model_path": str(model_path),
    }

    return metrics_row, y_pred


def main():
    parser = argparse.ArgumentParser(
        description="Train/test UniKP ExtraTreesRegressor with saved train/test features"
    )
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--n_jobs", type=int, default=-1)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = load_inputs_pkl(args.train_path, args.test_path)

    metrics_rows = []
    all_test_predictions = []

    for i in range(args.n_seeds):
        seed = args.seed + i
        metrics_row, y_pred = train_eval_one_seed(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            output_dir=output_dir,
            seed=seed,
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
        )
        metrics_rows.append(metrics_row)
        all_test_predictions.append(y_pred)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    pred_matrix = np.stack(all_test_predictions, axis=1)
    mean_pred = pred_matrix.mean(axis=1)
    std_pred = pred_matrix.std(axis=1)

    mean_pred_df = pd.DataFrame()
    mean_pred_df["label"] = y_test
    mean_pred_df["prediction_mean"] = mean_pred
    mean_pred_df["prediction_std"] = std_pred

    mean_metrics = compute_metrics(y_test, mean_pred)
    for key, value in mean_metrics.items():
        mean_pred_df[f"ensemble_{key}"] = value

    mean_pred_path = output_dir / "test_predictions_mean.csv"
    mean_pred_df.to_csv(mean_pred_path, index=False)

    summary = {
        "train_pkl": args.train_path,
        "test_pkl": args.test_path,
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
        "n_seeds": args.n_seeds,
        "seed_start": args.seed,
        "metrics_path": str(metrics_path),
        "mean_prediction_path": str(mean_pred_path),
        "metrics_mean_across_seeds": metrics_df[
            ["pcc", "scc", "mae", "rmse", "mse", "r2"]
        ].mean(numeric_only=True).to_dict(),
        "metrics_std_across_seeds": metrics_df[
            ["pcc", "scc", "mae", "rmse", "mse", "r2"]
        ].std(numeric_only=True).to_dict(),
        "ensemble_mean_prediction_metrics": mean_metrics,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(metrics_df)
    print("Mean prediction metrics:")
    print(json.dumps(mean_metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved mean test predictions to {mean_pred_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
