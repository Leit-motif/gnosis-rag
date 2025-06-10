#!/usr/bin/env python
"""
Embedding Pipeline Profiler
Comprehensive performance analysis tool for the embedding generation pipeline
"""
import os
import sys
import cProfile
import pstats
import io
import time
import psutil
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from backend.utils import load_config
from backend.rag_pipeline import RAGPipeline
from backend.fast_indexer import FastIndexer
from backend.obsidian_loader_v2 import ObsidianLoaderV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('profiling.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance measurements"""
    total_documents: int
    processing_time: float
    documents_per_second: float
    embeddings_per_second: float
    
    # Resource usage
    peak_memory_mb: float
    avg_cpu_percent: float
    peak_cpu_percent: float
    
    # Component timings
    document_loading_time: float
    preprocessing_time: float
    embedding_generation_time: float
    index_creation_time: float
    storage_time: float
    
    # API/Model specific
    api_call_count: int
    avg_api_latency: float
    batch_size_used: int
    concurrent_requests: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class SystemMonitor:
    """Monitor system resources during profiling"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.measurements = []
        self.monitoring = False
    
    def start_monitoring(self):
        """Start system resource monitoring"""
        self.monitoring = True
        self.measurements = []
    
    def stop_monitoring(self):
        """Stop monitoring and return summary stats"""
        self.monitoring = False
        if not self.measurements:
            return {}
        
        cpu_values = [m['cpu_percent'] for m in self.measurements]
        memory_values = [m['memory_mb'] for m in self.measurements]
        
        return {
            'peak_memory_mb': max(memory_values),
            'avg_memory_mb': np.mean(memory_values),
            'peak_cpu_percent': max(cpu_values),
            'avg_cpu_percent': np.mean(cpu_values),
            'sample_count': len(self.measurements)
        }
    
    def record_snapshot(self):
        """Record current system state"""
        if not self.monitoring:
            return
        
        try:
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent()
            
            self.measurements.append({
                'timestamp': time.time(),
                'memory_mb': memory_info.rss / 1024 / 1024,
                'cpu_percent': cpu_percent
            })
        except Exception as e:
            logger.warning(f"Failed to record system snapshot: {e}")

@contextmanager
def profile_function(name: str):
    """Context manager for profiling individual functions"""
    profiler = cProfile.Profile()
    start_time = time.time()
    
    logger.info(f"Starting profiling: {name}")
    profiler.enable()
    
    try:
        yield profiler
    finally:
        profiler.disable()
        elapsed_time = time.time() - start_time
        logger.info(f"Completed profiling: {name} in {elapsed_time:.2f}s")

class EmbeddingPipelineProfiler:
    """Main profiler class for embedding pipeline analysis"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else load_config()
        self.monitor = SystemMonitor()
        self.profile_results = {}
        
        # Create results directory
        self.results_dir = Path("profiling_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Profiler initialized, results will be saved to {self.results_dir}")
    
    def profile_document_loading(self, max_docs: int = 1000) -> Tuple[List[Dict], float]:
        """Profile document loading performance"""
        logger.info(f"Profiling document loading (max {max_docs} docs)")
        
        vault_loader = ObsidianLoaderV2(self.config["vault"]["path"])
        
        with profile_function("document_loading") as profiler:
            self.monitor.start_monitoring()
            start_time = time.time()
            
            # Load documents with monitoring
            documents = vault_loader.load_all_documents(config=self.config)
            
            # Limit documents for profiling
            if len(documents) > max_docs:
                documents = documents[:max_docs]
                logger.info(f"Limited to {max_docs} documents for profiling")
            
            loading_time = time.time() - start_time
            system_stats = self.monitor.stop_monitoring()
        
        # Save profiling stats
        self._save_profile_stats(profiler, "document_loading")
        
        # Prepare documents for indexing
        indexed_documents = [
            {
                'id': f"{doc.metadata.get('source', 'unknown_source')}#{doc.metadata.get('chunk_id', 'unknown_chunk')}", 
                'content': doc.page_content,
                'metadata': doc.metadata
            }
            for doc in documents
        ]
        
        logger.info(f"Loaded {len(indexed_documents)} documents in {loading_time:.2f}s")
        return indexed_documents, loading_time
    
    def profile_standard_pipeline(self, documents: List[Dict]) -> PerformanceMetrics:
        """Profile the standard RAG pipeline"""
        logger.info("Profiling standard RAG pipeline")
        
        rag_pipeline = RAGPipeline(self.config)
        
        with profile_function("standard_pipeline") as profiler:
            self.monitor.start_monitoring()
            start_time = time.time()
            
            # Profile the indexing process
            rag_pipeline.index_documents(documents)
            
            total_time = time.time() - start_time
            system_stats = self.monitor.stop_monitoring()
        
        # Save profiling stats
        self._save_profile_stats(profiler, "standard_pipeline")
        
        # Calculate metrics
        metrics = PerformanceMetrics(
            total_documents=len(documents),
            processing_time=total_time,
            documents_per_second=len(documents) / total_time if total_time > 0 else 0,
            embeddings_per_second=len(documents) / total_time if total_time > 0 else 0,
            peak_memory_mb=system_stats.get('peak_memory_mb', 0),
            avg_cpu_percent=system_stats.get('avg_cpu_percent', 0),
            peak_cpu_percent=system_stats.get('peak_cpu_percent', 0),
            document_loading_time=0,  # Measured separately
            preprocessing_time=0,  # Will be extracted from profile
            embedding_generation_time=0,  # Will be extracted from profile
            index_creation_time=0,  # Will be extracted from profile
            storage_time=0,  # Will be extracted from profile
            api_call_count=0,  # Will be extracted from logs
            avg_api_latency=0,  # Will be extracted from logs
            batch_size_used=rag_pipeline.batch_size,
            concurrent_requests=4  # ThreadPoolExecutor default
        )
        
        logger.info(f"Standard pipeline: {metrics.documents_per_second:.2f} docs/sec")
        return metrics
    
    def profile_fast_pipeline(self, documents: List[Dict]) -> PerformanceMetrics:
        """Profile the fast indexing pipeline"""
        logger.info("Profiling fast indexing pipeline")
        
        fast_indexer = FastIndexer(self.config)
        
        with profile_function("fast_pipeline") as profiler:
            self.monitor.start_monitoring()
            start_time = time.time()
            
            # Profile the fast indexing process
            import asyncio
            result = asyncio.run(fast_indexer.index_documents_fast(documents, resume=False))
            
            total_time = time.time() - start_time
            system_stats = self.monitor.stop_monitoring()
        
        # Save profiling stats
        self._save_profile_stats(profiler, "fast_pipeline")
        
        # Calculate metrics
        metrics = PerformanceMetrics(
            total_documents=len(documents),
            processing_time=total_time,
            documents_per_second=len(documents) / total_time if total_time > 0 else 0,
            embeddings_per_second=len(documents) / total_time if total_time > 0 else 0,
            peak_memory_mb=system_stats.get('peak_memory_mb', 0),
            avg_cpu_percent=system_stats.get('avg_cpu_percent', 0),
            peak_cpu_percent=system_stats.get('peak_cpu_percent', 0),
            document_loading_time=0,  # Measured separately
            preprocessing_time=0,  # Will be extracted from profile
            embedding_generation_time=0,  # Will be extracted from profile
            index_creation_time=0,  # Will be extracted from profile
            storage_time=0,  # Will be extracted from profile
            api_call_count=0,  # Will be extracted from logs
            avg_api_latency=0,  # Will be extracted from logs
            batch_size_used=fast_indexer.batch_size,
            concurrent_requests=fast_indexer.max_concurrent_requests
        )
        
        logger.info(f"Fast pipeline: {metrics.documents_per_second:.2f} docs/sec")
        return metrics
    
    def _save_profile_stats(self, profiler: cProfile.Profile, name: str):
        """Save profiling statistics to files"""
        # Save detailed stats
        stats_buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_buffer)
        stats.sort_stats('cumulative')
        stats.print_stats()
        
        stats_file = self.results_dir / f"{name}_profile_stats.txt"
        with open(stats_file, 'w') as f:
            f.write(stats_buffer.getvalue())
        
        # Save top functions as JSON for analysis
        stats.sort_stats('cumulative')
        top_functions = []
        for func, data in stats.stats.items():
            top_functions.append({
                'function': f"{func[0]}:{func[1]}({func[2]})",
                'cumulative_time': data[3],
                'calls': data[0],
                'per_call': data[3] / data[0] if data[0] > 0 else 0
            })
        
        # Sort by cumulative time and take top 20
        top_functions.sort(key=lambda x: x['cumulative_time'], reverse=True)
        top_functions = top_functions[:20]
        
        json_file = self.results_dir / f"{name}_top_functions.json"
        with open(json_file, 'w') as f:
            json.dump(top_functions, f, indent=2)
        
        logger.info(f"Saved profiling stats to {stats_file} and {json_file}")
    
    def create_performance_report(self, standard_metrics: PerformanceMetrics, 
                                 fast_metrics: PerformanceMetrics,
                                 loading_time: float) -> Dict[str, Any]:
        """Create a comprehensive performance report"""
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024**3,
                "python_version": sys.version
            },
            "document_loading": {
                "time_seconds": loading_time,
                "documents_loaded": standard_metrics.total_documents
            },
            "standard_pipeline": standard_metrics.to_dict(),
            "fast_pipeline": fast_metrics.to_dict(),
            "comparison": {
                "speed_improvement_factor": fast_metrics.documents_per_second / standard_metrics.documents_per_second if standard_metrics.documents_per_second > 0 else 0,
                "memory_usage_difference_mb": fast_metrics.peak_memory_mb - standard_metrics.peak_memory_mb,
                "time_reduction_percent": ((standard_metrics.processing_time - fast_metrics.processing_time) / standard_metrics.processing_time) * 100 if standard_metrics.processing_time > 0 else 0
            },
            "bottleneck_analysis": self._analyze_bottlenecks(),
            "recommendations": self._generate_recommendations(standard_metrics, fast_metrics)
        }
        
        # Save report
        report_file = self.results_dir / "performance_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved to {report_file}")
        return report
    
    def _analyze_bottlenecks(self) -> Dict[str, Any]:
        """Analyze profiling data to identify bottlenecks"""
        bottlenecks = {
            "top_time_consumers": [],
            "memory_intensive_operations": [],
            "api_performance": {},
            "recommendations": []
        }
        
        # Analyze each pipeline's profiling data
        for pipeline_name in ["standard_pipeline", "fast_pipeline"]:
            json_file = self.results_dir / f"{pipeline_name}_top_functions.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    top_functions = json.load(f)
                
                bottlenecks["top_time_consumers"].append({
                    "pipeline": pipeline_name,
                    "functions": top_functions[:5]  # Top 5 functions
                })
        
        return bottlenecks
    
    def _generate_recommendations(self, standard: PerformanceMetrics, 
                                 fast: PerformanceMetrics) -> List[str]:
        """Generate optimization recommendations based on profiling data"""
        recommendations = []
        
        # Performance comparison
        if fast.documents_per_second > standard.documents_per_second:
            improvement = (fast.documents_per_second / standard.documents_per_second - 1) * 100
            recommendations.append(f"Fast pipeline shows {improvement:.1f}% speed improvement - consider using it as default")
        
        # Memory usage analysis
        if fast.peak_memory_mb > standard.peak_memory_mb * 1.5:
            recommendations.append("Fast pipeline uses significantly more memory - consider reducing batch size for memory-constrained systems")
        
        # Throughput analysis
        if fast.documents_per_second < 50:  # Less than 50 docs/sec
            recommendations.append("Overall throughput is low - consider GPU acceleration or local embedding models")
        
        # Batch size optimization
        if fast.batch_size_used < 50:
            recommendations.append("Batch size is conservative - try increasing for better API efficiency")
        elif fast.batch_size_used > 150:
            recommendations.append("Large batch size may cause API rate limits - consider reducing if encountering errors")
        
        # Concurrency optimization
        if fast.concurrent_requests < 10:
            recommendations.append("Low concurrency - increase concurrent requests if API limits allow")
        
        return recommendations
    
    def create_visualization(self, report: Dict[str, Any]):
        """Create performance visualization charts"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Performance comparison
        pipelines = ['Standard', 'Fast']
        speeds = [report['standard_pipeline']['documents_per_second'], 
                 report['fast_pipeline']['documents_per_second']]
        
        axes[0, 0].bar(pipelines, speeds, color=['blue', 'orange'])
        axes[0, 0].set_title('Documents per Second Comparison')
        axes[0, 0].set_ylabel('Documents/Second')
        
        # Memory usage comparison
        memory = [report['standard_pipeline']['peak_memory_mb'], 
                 report['fast_pipeline']['peak_memory_mb']]
        
        axes[0, 1].bar(pipelines, memory, color=['blue', 'orange'])
        axes[0, 1].set_title('Peak Memory Usage Comparison')
        axes[0, 1].set_ylabel('Memory (MB)')
        
        # Processing time comparison
        times = [report['standard_pipeline']['processing_time'], 
                report['fast_pipeline']['processing_time']]
        
        axes[1, 0].bar(pipelines, times, color=['blue', 'orange'])
        axes[1, 0].set_title('Total Processing Time Comparison')
        axes[1, 0].set_ylabel('Time (seconds)')
        
        # CPU usage comparison
        cpu = [report['standard_pipeline']['avg_cpu_percent'], 
               report['fast_pipeline']['avg_cpu_percent']]
        
        axes[1, 1].bar(pipelines, cpu, color=['blue', 'orange'])
        axes[1, 1].set_title('Average CPU Usage Comparison')
        axes[1, 1].set_ylabel('CPU Percentage')
        
        plt.tight_layout()
        chart_file = self.results_dir / "performance_comparison.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance visualization saved to {chart_file}")
    
    def run_full_profile(self, max_docs: int = 1000) -> Dict[str, Any]:
        """Run complete profiling analysis"""
        logger.info(f"Starting full profiling analysis with max {max_docs} documents")
        
        try:
            # Profile document loading
            documents, loading_time = self.profile_document_loading(max_docs)
            
            # Profile standard pipeline
            logger.info("Profiling standard pipeline...")
            standard_metrics = self.profile_standard_pipeline(documents)
            
            # Profile fast pipeline
            logger.info("Profiling fast pipeline...")
            fast_metrics = self.profile_fast_pipeline(documents)
            
            # Create comprehensive report
            report = self.create_performance_report(standard_metrics, fast_metrics, loading_time)
            
            # Create visualizations
            self.create_visualization(report)
            
            # Print summary
            self._print_summary(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Profiling failed: {e}")
            raise
    
    def _print_summary(self, report: Dict[str, Any]):
        """Print a summary of profiling results"""
        print("\n" + "="*80)
        print("EMBEDDING PIPELINE PERFORMANCE ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\nDocument Loading:")
        print(f"  Documents processed: {report['document_loading']['documents_loaded']}")
        print(f"  Loading time: {report['document_loading']['time_seconds']:.2f} seconds")
        
        print(f"\nStandard Pipeline:")
        std = report['standard_pipeline']
        print(f"  Processing time: {std['processing_time']:.2f} seconds")
        print(f"  Throughput: {std['documents_per_second']:.2f} docs/sec")
        print(f"  Peak memory: {std['peak_memory_mb']:.1f} MB")
        print(f"  Batch size: {std['batch_size_used']}")
        
        print(f"\nFast Pipeline:")
        fast = report['fast_pipeline']
        print(f"  Processing time: {fast['processing_time']:.2f} seconds")
        print(f"  Throughput: {fast['documents_per_second']:.2f} docs/sec")
        print(f"  Peak memory: {fast['peak_memory_mb']:.1f} MB")
        print(f"  Batch size: {fast['batch_size_used']}")
        print(f"  Concurrent requests: {fast['concurrent_requests']}")
        
        print(f"\nComparison:")
        comp = report['comparison']
        print(f"  Speed improvement: {comp['speed_improvement_factor']:.2f}x")
        print(f"  Time reduction: {comp['time_reduction_percent']:.1f}%")
        print(f"  Memory difference: {comp['memory_usage_difference_mb']:.1f} MB")
        
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
        print(f"\nDetailed results saved to: {self.results_dir}")
        print("="*80)

def main():
    """Main function to run profiling"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile embedding pipeline performance")
    parser.add_argument("--max-docs", type=int, default=1000, 
                       help="Maximum number of documents to profile (default: 1000)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file")
    
    args = parser.parse_args()
    
    try:
        profiler = EmbeddingPipelineProfiler(args.config)
        profiler.run_full_profile(args.max_docs)
        
    except KeyboardInterrupt:
        logger.info("Profiling interrupted by user")
    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 