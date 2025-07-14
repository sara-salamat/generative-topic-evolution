# Topic Evolution for Conferences

**🚧 UNDER ACTIVE DEVELOPMENT 🚧**

This project is an experimental platform for exploring and understanding the evolution of research topics in top academic conferences using Retrieval-Augmented Generation (RAG) techniques.

## 📊 Project Status

### ✅ **Completed Components**

#### 1. **Data Pipeline** 
- ✅ OpenReview data fetching (`scripts/fetch_openreview_data*.py`)
- ✅ Data cleaning and preprocessing (`notebooks/01_data_cleaning.ipynb`)
- ✅ Weaviate vector database integration
- ✅ SPECTER embeddings for paper similarity

#### 2. **Summarization System**
- ✅ Summarizer using Ollama/Qwen models (`scripts/summarizer_resume_fast.py`)
- ✅ Memory-efficient processing with progress tracking
- ✅ Batch processing with error handling and recovery
- ✅ Automatic garbage collection and memory management

#### 3. **RAG Engine** (`retrieval/`)
- ✅ Basic RAG system with Weaviate retriever
- ✅ Query parsing with year/topic extraction (`utils/query_parser.py`)
- ✅ LLM integration via Ollama API
- ✅ Vector similarity search with filtering

#### 4. **Topic Evolution Analyzer** (`analysis/`)
- ✅ **NEW**: Comprehensive topic evolution analysis engine
- ✅ Topic trend tracking over time periods
- ✅ Emerging topic detection and growth rate calculation
- ✅ Topic relationship analysis and centrality metrics
- ✅ LLM-powered topic extraction with TF-IDF fallback
- ✅ Command-line interface and Python API
- ✅ Automated visualization generation
- ✅ Comprehensive reporting system


### 🚧 **Remaining Work**

#### 1. **Web Application**
- ❌ Interactive web interface (Streamlit/FastAPI)
- ❌ Topic visualization dashboard
- ❌ Trend analysis interface
- ❌ Search and exploration tools

#### 2. **Enhanced RAG System**
- ❌ Better context building (implement TODO in `rag_engine.py`)
- ❌ Multi-hop reasoning capabilities
- ❌ Citation tracking and source attribution
- ❌ Confidence scoring for responses

#### 3. **Evaluation Framework**
- ❌ Summary quality metrics
- ❌ Topic clustering accuracy evaluation
- ❌ RAG response relevance scoring
- ❌ Trend prediction accuracy measurement

#### 4. **Advanced Visualizations**
- ❌ Interactive topic evolution timelines
- ❌ Topic relationship network graphs
- ❌ Citation network analysis
- ❌ Trend heatmaps and forecasting

#### 5. **Performance Optimization**
- ❌ Caching mechanisms
- ❌ Parallel processing
- ❌ Database query optimization
- ❌ Model performance tuning

#### 6. **Automated Data Ingestion**
- ❌ Automated pipeline for broader conference coverage
- ❌ Support for multiple conference sources
- ❌ Real-time data updates
- ❌ Conference-specific analysis capabilities

## 🚀 Project Vision

Our goal is to build a tool that helps understand how research topics evolve in academic conferences over time. The system aims to:

- **Summarize** large volumes of conference submissions efficiently
- **Track** how research topics change and evolve over years
- **Visualize** topic relationships and emerging research directions
- **Analyze** patterns across different conferences and time periods

**Future Plans**: Once the core tool is working, we plan to automate data ingestion to broaden coverage to any conference, enabling comprehensive cross-conference analysis and real-time topic evolution tracking.

## 🛠️ Current Architecture

```
generative-topic-evolution/
├── analysis/                    # ✅ Topic Evolution Analyzer
│   ├── topic_evolution_analyzer.py
│   ├── test_topic_analyzer.py
│   └── run_topic_analysis.py
├── retrieval/                   # ✅ RAG Engine
│   ├── rag_engine.py
│   ├── weaviate_retriever.py
│   └── retriever.py
├── scripts/                     # ✅ Data Processing
│   ├── summarizer_resume_fast.py
│   ├── fetch_openreview_data*.py
│   └── embed_abstracts.py
├── utils/                       # ✅ Utilities
│   └── query_parser.py
├── data/                        # ✅ Processed Data
│   └── processed/
├── notebooks/                   # ✅ Data Cleaning
│   └── 01_data_cleaning.ipynb
└── app/                         # ❌ Web Interface (TODO)
```



## 📈 Development Roadmap

### **Phase 1: Foundation** ✅ COMPLETED
- [x] Data pipeline and preprocessing
- [x] Basic RAG system
- [x] Summarization pipeline
- [x] Topic evolution analyzer

### **Phase 2: Web Interface**
- [ ] Interactive dashboard
- [ ] Topic visualization
- [ ] Search interface
- [ ] User authentication

### **Phase 3: Advanced Features**
- [ ] Enhanced RAG with multi-hop reasoning
- [ ] Evaluation framework
- [ ] Advanced visualizations
- [ ] Performance optimization

### **Phase 4: Production**
- [ ] Deployment infrastructure
- [ ] Monitoring and logging
- [ ] API documentation
- [ ] User guides

## 🚨 Important Notes

### **Development Status**
- **⚠️ EXPERIMENTAL**: This project is under active development
- **🧪 RESEARCH**: Results are experimental and may vary
- **📊 LIMITED TESTING**: Not yet production-ready

### **Current Limitations**
- Summarization is resource-intensive and may take days for large datasets
- RAG system is basic and needs enhancement
- No web interface yet
- Limited evaluation metrics
- Memory usage can be high with large datasets

## 🤝 Contributing

### **Areas Needing Help**
1. **Web Interface Development**
2. **RAG System Enhancement**
3. **Evaluation Framework**
4. **Performance Optimization**
5. **Documentation and Testing**

