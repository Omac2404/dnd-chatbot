"""
RAG sistemini değerlendir
Accuracy, response time, retrieval quality ölçüm
"""

import time
from typing import List, Dict
from rag_pipeline import RAGPipeline
from test_questions import TEST_QUESTIONS


class RAGEvaluator:
    """RAG evaluation sınıfı"""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag = rag_pipeline
        self.results = []
    
    def evaluate_answer_quality(self, answer: str, expected_keywords: List[str]) -> float:
        """
        Cevabın kalitesini expected keywords'e göre değerlendir
        
        Returns:
            0.0 - 1.0 arası skor
        """
        answer_lower = answer.lower()
        found_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
        
        return found_keywords / len(expected_keywords) if expected_keywords else 0.0
    
    def evaluate_retrieval_quality(self, question: str, retrieved_docs: List[Dict], expected_keywords: List[str]) -> float:
        """Retrieved document'ların kalitesini değerlendir"""
        
        # Retrieved text'leri birleştir
        if retrieved_docs and 'text_preview' in retrieved_docs[0]:
            all_retrieved_text = " ".join([doc['text_preview'] for doc in retrieved_docs]).lower()
        else:
            all_retrieved_text = " ".join([doc.get('text', '') for doc in retrieved_docs]).lower()
        
        # Expected keywords kaçı var?
        found_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in all_retrieved_text)
        
        # ÖNEMLİ: RETURN SATIRI OLMALI!
        return found_keywords / len(expected_keywords) if expected_keywords else 0.0
    
    def run_evaluation(self, test_questions: List[Dict] = None):
        """Tüm test sorularını çalıştır ve değerlendir"""
        
        if test_questions is None:
            test_questions = TEST_QUESTIONS
        
        print("="*60)
        print("RAG EVALUATION")
        print("="*60)
        print(f"Toplam {len(test_questions)} test sorusu\n")
        
        for i, test_case in enumerate(test_questions, 1):
            print(f"\n{'='*60}")
            print(f"TEST {i}/{len(test_questions)}: {test_case['category']}")
            print(f"{'='*60}")
            print(f"Soru: {test_case['question']}")
            
            # Timing
            start_time = time.time()
            
            # RAG query
            result = self.rag.query(test_case['question'], top_k=5)
            
            # Retrieval için ayrı timing
            retrieval_docs = self.rag.retrieve_context(test_case['question'], top_k=5)
            
            elapsed_time = time.time() - start_time
            
            # Evaluate
            answer_quality = self.evaluate_answer_quality(
                result['answer'], 
                test_case['expected_keywords']
            )
            
            retrieval_quality = self.evaluate_retrieval_quality(
                test_case['question'],
                result['sources'],
                test_case['expected_keywords']
            )
            
            confidence = self.rag.calculate_confidence(result['answer'])
            
            # Store result
            eval_result = {
                "question": test_case['question'],
                "category": test_case['category'],
                "answer": result['answer'],
                "answer_quality": answer_quality,
                "retrieval_quality": retrieval_quality,
                "confidence": confidence,
                "response_time": elapsed_time,
                "sources_used": len(result['sources'])
            }
            
            self.results.append(eval_result)
            
            # Print result
            print(f"\n📊 Skorlar:")
            print(f"   Answer Quality: {answer_quality:.2%}")
            print(f"   Retrieval Quality: {retrieval_quality:.2%}")
            print(f"   Confidence: {confidence:.2%}")
            print(f"   Response Time: {elapsed_time:.2f}s")
            
            # Cevabın kısa önizlemesi
            print(f"\n💬 Cevap Preview:")
            print(f"   {result['answer'][:200]}...")
        
        # Overall stats
        self.print_overall_stats()
    
    def print_overall_stats(self):
        """Genel istatistikleri yazdır"""
        
        if not self.results:
            print("\n❌ Henüz sonuç yok!")
            return
        
        print("\n" + "="*60)
        print("GENEL İSTATİSTİKLER")
        print("="*60)
        
        # Calculate averages
        avg_answer_quality = sum(r['answer_quality'] for r in self.results) / len(self.results)
        avg_retrieval_quality = sum(r['retrieval_quality'] for r in self.results) / len(self.results)
        avg_confidence = sum(r['confidence'] for r in self.results) / len(self.results)
        avg_response_time = sum(r['response_time'] for r in self.results) / len(self.results)
        
        print(f"\n📈 Ortalama Skorlar:")
        print(f"   Answer Quality: {avg_answer_quality:.2%}")
        print(f"   Retrieval Quality: {avg_retrieval_quality:.2%}")
        print(f"   Confidence: {avg_confidence:.2%}")
        print(f"   Response Time: {avg_response_time:.2f}s")
        
        # Category breakdown
        print(f"\n📊 Kategori Bazlı:")
        categories = {}
        for result in self.results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result['answer_quality'])
        
        for cat, scores in categories.items():
            avg_score = sum(scores) / len(scores)
            print(f"   {cat}: {avg_score:.2%} ({len(scores)} soru)")
        
        # Pass/Fail
        passed = sum(1 for r in self.results if r['answer_quality'] >= 0.6)
        total = len(self.results)
        
        print(f"\n✅ Başarı Oranı: {passed}/{total} ({passed/total:.1%})")
        
        if avg_answer_quality >= 0.75:
            print("\n🎉 RAG sistemi iyi performans gösteriyor!")
        elif avg_answer_quality >= 0.6:
            print("\n⚠️ RAG sistemi orta performans gösteriyor. İyileştirme yapılabilir.")
        else:
            print("\n❌ RAG sistemi zayıf performans gösteriyor. Optimizasyon gerekli!")


def main():
    """Evaluation test scripti"""
    
    print("="*60)
    print("RAG SYSTEM EVALUATION")
    print("="*60 + "\n")
    
    # RAG pipeline oluştur
    rag = RAGPipeline(use_local_llm=True)
    
    # Evaluator oluştur
    evaluator = RAGEvaluator(rag)
    
    # Evaluation çalıştır
    evaluator.run_evaluation()
    
    print("\n✅ Evaluation tamamlandı!")


if __name__ == "__main__":
    main()