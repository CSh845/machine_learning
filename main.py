import os
import argparse
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
from utils.data_loader import load_fingerprints, load_descriptors
from utils.trainer import train_model
from utils.evaluator import evaluate_model
from utils.saver import save_model, save_results_to_csv
from utils.plotter import plot_auc_curves
from models.models import get_all_models
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from joblib import dump


def parse_arguments():
    parser = argparse.ArgumentParser(description="主控制文件，用于训练模型并生成AUC曲线。")

    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        required=True,
        choices=['MLP', 'RandomForest', 'SVM', 'XGBoost'],
        help="要训练的模型列表。可选：MLP, RandomForest, SVM, XGBoost。"
    )

    parser.add_argument(
        '--fingerprints',
        type=str,
        nargs='*',
        default=[],
        choices=['MACCS', 'Morgan', 'RDKit', 'TopologicalTorsion', 'AtomPairsFP'],
        help="要使用的指纹类型。可以指定多个，用空格分隔。"
    )

    parser.add_argument(
        '--use_descriptors',
        action='store_true',
        help="是否使用分子描述符作为特征输入。"
    )

    parser.add_argument(
        '--descriptors',
        type=str,
        default='data/descriptors.csv',
        help="分子描述符文件路径。默认：'data/1descriptors.csv'。"
    )

    parser.add_argument(
        '--data',
        type=str,
        default='data/data.csv',
        help="输入数据CSV文件路径。默认：'data/data.csv'。"
    )

    parser.add_argument(
        '--n_splits',
        type=int,
        default=5,
        help="K-Fold交叉验证的折数。默认：5。"
    )

    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help="测试集所占比例。默认：0.2。"
    )

    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help="随机种子。默认：42。"
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default='saved_models',
        help="保存训练模型的目录。默认：'saved_models'。"
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help="保存评估结果和AUC曲线的目录。默认：'results'。"
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help="增加输出的详细程度。"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # 创建保存目录（如果不存在）
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # 获取所有模型
    models_dict = get_all_models()

    # 设置KFold
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    for model_name in args.models:
        if model_name not in models_dict:
            print(f"模型 '{model_name}' 未定义。可用模型: {list(models_dict.keys())}")
            continue

        # 调用模型函数以获取 (model, params)
        model_instance, params = models_dict[model_name]()
        print(f"\n=== 开始训练模型: {model_name} ===")

        auc_data = {}  # 用于存储每种指纹或描述符的AUC曲线数据
        results = []  # 用于存储评估指标

        # 训练指纹特征的模型（如果有指纹）
        if args.fingerprints:
            for fingerprint_type in args.fingerprints:
                print(f"\n训练模型: {model_name}，指纹类型: {fingerprint_type}")

                # 加载指纹特征
                X, y = load_fingerprints(args.data, fingerprint_type=fingerprint_type)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=args.test_size, random_state=args.random_state
                )

                # 训练模型
                grid_search = train_model(
                    X_train, y_train, model_instance, params, kf,
                    scoring='roc_auc',
                    verbose_level=2 if args.verbose else 0,
                    n_jobs=-1
                )
                if args.verbose:
                    print(f"最佳参数 for {fingerprint_type}: {grid_search.best_params_}")

                # 保存最佳模型
                best_model_filename = os.path.join(
                    args.save_dir, f'best_{model_name.lower()}_model_{fingerprint_type}.joblib'
                )
                save_model(grid_search.best_estimator_, best_model_filename)
                print(f"最佳模型已保存为: {best_model_filename}")

                # 评估模型
                metrics, (fpr, tpr), _ = evaluate_model(
                    grid_search.best_estimator_, X_test, y_test, model_type=model_name.lower()
                )
                metrics.update(grid_search.best_params_)
                metrics['Model'] = f'{fingerprint_type}-{model_name}'
                results.append(metrics)

                # 存储AUC曲线数据
                auc_data[fingerprint_type] = (fpr, tpr)

        # 训练描述符特征的模型（如果指定）
        if args.use_descriptors:
            print(f"\n训练模型: {model_name}，特征类型: Descriptors")

            # 加载描述符特征，注意这里将 scale 设置为 False，以避免重复标准化
            X, y = load_descriptors(args.descriptors, scale=False)

            # 拆分数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, random_state=args.random_state
            )

            # 填补缺失值
            imputer = SimpleImputer(strategy='mean')
            X_train_imputed = imputer.fit_transform(X_train)
            X_test_imputed = imputer.transform(X_test)

            # 标准化数据
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imputed)
            X_test_scaled = scaler.transform(X_test_imputed)

            # 重新实例化模型以避免参数共享
            model_instance_desc, params_desc = models_dict[model_name]()

            # 训练模型
            grid_search = train_model(
                X_train_scaled, y_train, model_instance_desc, params_desc, kf,
                scoring='roc_auc',
                verbose_level=2 if args.verbose else 0,
                n_jobs=-1
            )
            if args.verbose:
                print(f"最佳参数 for Descriptors: {grid_search.best_params_}")

            # 保存最佳模型
            best_model_filename = os.path.join(
                args.save_dir, f'best_{model_name.lower()}_model_Descriptors.joblib'
            )
            save_model(grid_search.best_estimator_, best_model_filename)
            print(f"最佳模型已保存为: {best_model_filename}")

            # 保存 imputer 和 scaler
            imputer_filename = os.path.join(args.save_dir, f'{model_name.lower()}_imputer.joblib')
            scaler_filename = os.path.join(args.save_dir, f'{model_name.lower()}_scaler.joblib')

            dump(imputer, imputer_filename)
            dump(scaler, scaler_filename)

            print(f"Imputer 已保存到 {imputer_filename}")
            print(f"Scaler 已保存到 {scaler_filename}")

            # 评估模型
            metrics, (fpr, tpr), _ = evaluate_model(
                grid_search.best_estimator_, X_test_scaled, y_test, model_type=model_name.lower()
            )
            metrics.update(grid_search.best_params_)
            metrics['Model'] = f'Descriptors-{model_name}'
            results.append(metrics)

            # 存储AUC曲线数据
            auc_data['Descriptors'] = (fpr, tpr)

        # 保存评估结果
        results_df = pd.DataFrame(results)
        results_filename = os.path.join(args.results_dir, f'{model_name}_combined_results.csv')
        save_results_to_csv(results_df, results_filename)
        print(f"\n所有评估结果已保存到 '{results_filename}'")

        # 绘制并保存AUC曲线
        plot_auc_curves(auc_data, model_name, args.results_dir)

        # 保存AUC曲线数据到CSV
        auc_csv_filename = os.path.join(args.results_dir, f'{model_name}_auc_data.csv')
        auc_records = []
        for fingerprint, (fpr, tpr) in auc_data.items():
            for f, t in zip(fpr, tpr):
                auc_records.append({
                    'Fingerprint': fingerprint,
                    'FPR': f,
                    'TPR': t
                })
        auc_df = pd.DataFrame(auc_records)
        auc_df.to_csv(auc_csv_filename, index=False)
        print(f"AUC曲线数据已保存到 '{auc_csv_filename}'")


if __name__ == "__main__":
    main()
