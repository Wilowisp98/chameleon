import os
import torch
import matplotlib.pyplot as plt
import argparse

def analyze_model_metrics(model_name):
    model_path = os.path.join('unet/model/', model_name)
    try:
        checkpoint = torch.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    best_metrics = {
        'pixel_acc': {'epoch': 0, 'value': float('-inf')},
        'mean_dice_coeff': {'epoch': 0.0, 'value': float('-inf')},
        'combined': {'epoch': 0.0, 'value': float('-inf')}
    }

    metrics = checkpoint['metrics']
    
    for epoch_counter, epoch in enumerate(metrics):
        if epoch['pixel_acc'] > best_metrics['pixel_acc']['value']:
            best_metrics['pixel_acc']['value'] = epoch['pixel_acc']
            best_metrics['pixel_acc']['epoch'] = epoch_counter

        if epoch['mean_dice_coeff'] > best_metrics['mean_dice_coeff']['value']:
            best_metrics['mean_dice_coeff']['value'] = epoch['mean_dice_coeff']
            best_metrics['mean_dice_coeff']['epoch'] = epoch_counter
            
        if epoch['pixel_acc'] + epoch['mean_dice_coeff'] > best_metrics['combined']['value']:
            best_metrics['combined']['value'] = epoch['pixel_acc'] + epoch['mean_dice_coeff']
            best_metrics['combined']['epoch'] = epoch_counter

    print("\nBest Metrics Results:")
    print(f"Best Pixel Accuracy: {best_metrics['pixel_acc']['value']:.4f} (Epoch {best_metrics['pixel_acc']['epoch']})")
    print(f"Best Mean Dice Coefficient: {best_metrics['mean_dice_coeff']['value']:.4f} (Epoch {best_metrics['mean_dice_coeff']['epoch']})")
    print(f"Best Combined Score: {best_metrics['combined']['value']:.4f} (Epoch {best_metrics['combined']['epoch']})")

    epochs = range(len(metrics))
    plt.style.use('seaborn')
    plt.rcParams['figure.dpi'] = 300

    os.makedirs('model_results', exist_ok=True)

    plots = {
        'global_metrics.png': {
            'title': 'Global Metrics Over Training',
            'metrics': [
                ('global_acc', 'Accuracy', 'o'),
                ('global_precision', 'Precision', 's'),
                ('global_recall', 'Recall', '^'),
                ('global_f1', 'F1-Score', 'D')
            ]
        },
        'mean_metrics.png': {
            'title': 'Mean Metrics Over Training',
            'metrics': [
                ('mean_acc', 'Accuracy', 'o'),
                ('mean_precision', 'Precision', 's'),
                ('mean_recall', 'Recall', '^'),
                ('mean_f1', 'F1-Score', 'D')
            ]
        },
        'precision_comparison.png': {
            'title': 'Precision Metrics Comparison',
            'metrics': [
                ('global_precision', 'Global Precision', 'o'),
                ('mean_precision', 'Mean Precision', 's')
            ]
        },
        'recall_comparison.png': {
            'title': 'Recall Metrics Comparison',
            'metrics': [
                ('global_recall', 'Global Recall', 'o'),
                ('mean_recall', 'Mean Recall', 's')
            ]
        },
        'f1_comparison.png': {
            'title': 'F1 Score Comparison',
            'metrics': [
                ('global_f1', 'Global F1', 'o'),
                ('mean_f1', 'Mean F1', 's')
            ]
        },
        'dice_coefficient.png': {
            'title': 'Mean Dice Coefficient Over Training',
            'metrics': [('mean_dice_coeff', None, 'o')]
        },
        'pixel_accuracy.png': {
            'title': 'Pixel Accuracy Over Training',
            'metrics': [('pixel_acc', None, 'o')]
        },
        'average_loss.png': {
            'title': 'Average Loss Over Training',
            'metrics': [('average_loss', None, 'o')]
        }
    }

    for filename, plot_info in plots.items():
        plt.figure(figsize=(10, 6))
        for metric_name, label, marker in plot_info['metrics']:
            values = [m[metric_name] for m in metrics]
            if metric_name == 'average_loss':
                values = [v.detach().numpy() for v in values]
            plt.plot(epochs, values, label=label, marker=marker)
        
        plt.title(plot_info['title'])
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        if any(m[1] for m in plot_info['metrics']):
            plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join('model_results', filename), bbox_inches='tight')
        plt.close()

    print(f"\nPlots have been saved in the 'metric_plots' directory")

def main():
    parser = argparse.ArgumentParser(description='Analyze model metrics and generate plots')
    parser.add_argument('model_name', type=str, help='Name of the model checkpoint file')
    args = parser.parse_args()
    
    analyze_model_metrics(args.model_name)

if __name__ == '__main__':
    main()