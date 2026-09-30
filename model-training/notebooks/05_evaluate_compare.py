import os, sys, time, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil

# Directories
_model_training_dir = Path(__file__).resolve().parent.parent
_models_dir = _model_training_dir / "models"
_results_dir = _model_training_dir / "results"
_dataset_test_dir = _model_training_dir / "dataset" / "test"
_live_models_dir = _model_training_dir.parent / "models"

_results_dir.mkdir(exist_ok=True)
_live_models_dir.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load test set
print("=" * 60)
print("STEP 1: LOAD TEST SET")
print("=" * 60)
test_ds = tf.keras.utils.image_dataset_from_directory(
    str(_dataset_test_dir),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)
class_names = test_ds.class_names
print("Class names from test set:", class_names)
print("Total test images:", sum(1 for _ in test_ds.unbatch()))

# Load class indices
with open(str(_models_dir / "class_indices.json"), "r") as f:
    class_indices = json.load(f)
print("Class indices from JSON:", class_indices)

# Verify class order
expected_order = [name for name, idx in sorted(class_indices.items(), key=lambda x: x[1])]
print("Expected order from class_indices.json:", expected_order)
print("Dataset class order:", class_names)
if class_names == expected_order:
    print("[OK] Class order matches class_indices.json")
else:
    print("[WARNING] Class order MISMATCH - metrics may be corrupted")
    print("  Expected:", expected_order)
    print("  Got:     ", class_names)

test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)
print()

# Models to evaluate
# NOTE: Neither model was trained with model-specific preprocess_input.
# Both were trained on raw [0, 255] pixel values from image_dataset_from_directory.
# Applying preprocess_input here would break inference, especially for MobileNetV2.
models_info = [
    {
        "name": "mobilenetv2",
        "display": "MobileNetV2",
        "path": _models_dir / "mobilenetv2_final.keras",
    },
    {
        "name": "vgg16",
        "display": "VGG16",
        "path": _models_dir / "vgg16_final.keras",
    },
]

# Get all test images as numpy arrays for inference
print("Loading all test images into memory...")
all_images = []
all_labels = []
for images, labels in test_ds.unbatch():
    all_images.append(images.numpy())
    all_labels.append(labels.numpy())
all_images = np.array(all_images)
all_labels = np.array(all_labels)
all_label_indices = np.argmax(all_labels, axis=1)
print("Loaded", len(all_images), "test images, shape:", all_images.shape)
print()

results_rows = []

for info in models_info:
    model_name = info["name"]
    display_name = info["display"]
    model_path = info["path"]

    print("=" * 60)
    print("EVALUATING:", display_name)
    print("=" * 60)

    # Load model
    model = tf.keras.models.load_model(str(model_path))
    print("[OK] Model loaded from:", model_path)

    # Model size
    model_size_mb = model_path.stat().st_size / 1024 / 1024
    total_params = model.count_params()
    trainable_params = sum(p.numpy().size for p in model.trainable_weights)
    print("Model size:", round(model_size_mb, 1), "MB")
    print("Total params:", f"{total_params:,}")
    print("Trainable params:", f"{trainable_params:,}")

    # Predict (models were trained on raw [0,255] pixels, no preprocess_input needed)
    predictions = model.predict(all_images, batch_size=BATCH_SIZE, verbose=1)
    pred_indices = np.argmax(predictions, axis=1)

    # Overall accuracy
    overall_acc = np.mean(pred_indices == all_label_indices)
    print()
    print("Overall test accuracy:", round(overall_acc, 4), "(", round(overall_acc * 100, 1), "%)")

    # Classification report
    print()
    print("Classification Report:")
    report = classification_report(
        all_label_indices, pred_indices,
        target_names=expected_order,
        digits=4,
        zero_division=0
    )
    print(report)

    # Extract macro and weighted F1
    report_dict = classification_report(
        all_label_indices, pred_indices,
        target_names=expected_order,
        output_dict=True,
        zero_division=0
    )
    macro_f1 = report_dict["macro avg"]["f1-score"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    # Confusion matrix
    cm = confusion_matrix(all_label_indices, pred_indices)
    print()
    print("Confusion matrix:")
    print(cm)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=expected_order,
        yticklabels=expected_order,
        title=display_name + " -- Test Set Confusion Matrix",
        ylabel='True',
        xlabel='Predicted'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    plt.colorbar(im)
    plt.tight_layout()
    cm_path = _results_dir / (model_name + "_test_confusion_matrix.png")
    plt.savefig(str(cm_path), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Confusion matrix saved to:", cm_path)

    # Inference speed benchmark
    print()
    print("Inference speed benchmark (CPU, single image)...")
    num_benchmark = min(30, len(all_images))
    benchmark_indices = np.linspace(0, len(all_images) - 1, num_benchmark, dtype=int)
    times = []
    for idx in benchmark_indices:
        single_img = all_images[idx:idx+1]
        t0 = time.perf_counter()
        _ = model.predict(single_img, verbose=0)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    avg_ms = np.mean(times)
    throughput = 1000.0 / avg_ms if avg_ms > 0 else 0
    print("Avg inference time:", round(avg_ms, 2), "ms per image (CPU)")
    print("Throughput:", round(throughput, 1), "images/sec (CPU)")

    results_rows.append({
        "Model": display_name,
        "Test Accuracy": round(overall_acc, 4),
        "Macro F1": round(macro_f1, 4),
        "Weighted F1": round(weighted_f1, 4),
        "Avg Inference (ms)": round(avg_ms, 2),
        "Throughput (img/sec)": round(throughput, 1),
        "Model Size (MB)": round(model_size_mb, 1),
        "Total Params": total_params,
    })

    print()
    print()

# Comparison table
print("=" * 60)
print("MODEL COMPARISON TABLE")
print("=" * 60)
try:
    import pandas as pd
    df = pd.DataFrame(results_rows)
    print(df.to_string(index=False))
    csv_path = _results_dir / "model_comparison.csv"
    df.to_csv(str(csv_path), index=False)
    print()
    print("[OK] Comparison CSV saved to:", csv_path)
except ImportError:
    print("[WARNING] pandas not installed, printing raw table")
    csv_path = _results_dir / "model_comparison.csv"
    headers = list(results_rows[0].keys())
    with open(str(csv_path), "w") as f:
        f.write(",".join(headers) + "\n")
        for row in results_rows:
            f.write(",".join(str(row[h]) for h in headers) + "\n")
    print("[OK] Comparison CSV saved to:", csv_path)

# Automatic recommendation
print()
print("=" * 60)
print("AUTOMATIC RECOMMENDATION")
print("=" * 60)

mobilenet_row = [r for r in results_rows if r["Model"] == "MobileNetV2"][0]
vgg16_row = [r for r in results_rows if r["Model"] == "VGG16"][0]

acc_diff = vgg16_row["Test Accuracy"] - mobilenet_row["Test Accuracy"]
speed_ratio = vgg16_row["Avg Inference (ms)"] / mobilenet_row["Avg Inference (ms)"]
size_ratio = vgg16_row["Model Size (MB)"] / mobilenet_row["Model Size (MB)"]

print("Accuracy gap:", round(acc_diff * 100, 1), "percentage points (VGG16 advantage)")
print("Inference time ratio:", round(speed_ratio, 1), "x slower (VGG16 vs MobileNetV2)")
print("Size ratio:", round(size_ratio, 1), "x larger (VGG16 vs MobileNetV2)")
print()

if acc_diff > 0.10 and speed_ratio < 3.0:
    winner = "vgg16"
    winner_display = "VGG16"
    reasoning = (
        "VGG16 is recommended because its accuracy advantage ("
        + str(round(acc_diff * 100, 1))
        + " percentage points) justifies the moderate increase in inference time ("
        + str(round(speed_ratio, 1)) + "x). For a real-time web application, "
        + "the extra latency is acceptable when users expect accurate diagnostics."
    )
elif acc_diff > 0.05:
    winner = "mobilenetv2"
    winner_display = "MobileNetV2"
    reasoning = (
        "MobileNetV2 is recommended. The accuracy gap ("
        + str(round(acc_diff * 100, 1))
        + " percentage points) is modest, while MobileNetV2 is "
        + str(round(speed_ratio, 1)) + "x faster and "
        + str(round(size_ratio, 1)) + "x smaller. For a real-time web app "
        + "where users upload photos and wait for results, the faster response "
        + "time provides a better user experience and lower infrastructure costs."
    )
else:
    winner = "mobilenetv2"
    winner_display = "MobileNetV2"
    reasoning = (
        "MobileNetV2 is recommended. The accuracy gap is negligible ("
        + str(round(acc_diff * 100, 1))
        + " percentage points), while MobileNetV2 is significantly faster and "
        + "smaller. For a real-time web application, the lower latency and "
        + "resource usage make it the clear choice."
    )

print("Reasoning:", reasoning)
print()

# Export winner to live service
winner_path = _models_dir / (winner + "_final.keras")
dest_model = _live_models_dir / "production_model.keras"
dest_indices = _live_models_dir / "production_class_indices.json"

shutil.copy2(str(winner_path), str(dest_model))
shutil.copy2(str(_models_dir / "class_indices.json"), str(dest_indices))

print("=" * 60)
print("WINNER EXPORTED TO LIVE SERVICE")
print("=" * 60)
print("Copied model to:", dest_model.resolve())
print("Copied class indices to:", dest_indices.resolve())
print()

# Final summary
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print("Selected model for deployment:", winner_display)
print("Reason:", reasoning)
print()
print("Restart the FastAPI service:")
print("  uvicorn app.main:app --reload --port 8000")
