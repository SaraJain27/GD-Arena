from sentence_transformers import SentenceTransformer, util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch

# =========================
# SETUP
# =========================
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
vader = SentimentIntensityAnalyzer()

# =========================
# RELEVANCE SCORE (SBERT)
# =========================
def get_relevance_score(topic, text):
    topic_embedding = sbert_model.encode(topic, convert_to_tensor=True)
    text_embedding = sbert_model.encode(text, convert_to_tensor=True)
    similarity = util.cos_sim(topic_embedding, text_embedding)
    score = float(similarity[0][0]) * 100
    return round(score, 2)

# =========================
# CONFIDENCE SCORE (VADER)
# =========================
def get_confidence_score(text):
    scores = vader.polarity_scores(text)
    confidence = (scores['compound'] + 1) / 2 * 100
    return round(confidence, 2)

# =========================
# UNIQUE POINTS (SBERT)
# =========================
def get_unique_points(segments):
    if len(segments) == 0:
        return 0
    
    embeddings = sbert_model.encode(segments, convert_to_tensor=True)
    unique_count = 1
    
    for i in range(1, len(embeddings)):
        is_unique = True
        for j in range(i):
            similarity = float(util.cos_sim(embeddings[i], embeddings[j]))
            if similarity > 0.85:
                is_unique = False
                break
        if is_unique:
            unique_count += 1
    
    return unique_count

# =========================
# FINAL SCORING
# =========================
def score_debaters(topic, diarization_result):
    # Group segments by speaker
    speaker_segments = {}
    
    for seg in diarization_result["segments"]:
        speaker = seg.get("speaker", "unknown")
        text = seg["text"].strip()
        duration = seg["end"] - seg["start"]
        
        if speaker not in speaker_segments:
            speaker_segments[speaker] = {
                "texts": [],
                "total_time": 0
            }
        
        speaker_segments[speaker]["texts"].append(text)
        speaker_segments[speaker]["total_time"] += duration
    
    # Calculate total talk time
    total_time = sum(s["total_time"] for s in speaker_segments.values())
    
    # Score each speaker
    results = {}
    speaker_count = 0
    
    for speaker, data in speaker_segments.items():
        name = f"Debater {chr(65 + speaker_count)}"
        speaker_count += 1
        
        full_text = " ".join(data["texts"])
        
        # 1. Relevance Score (40%)
        relevance = get_relevance_score(topic, full_text)
        
        # 2. Confidence Score (15%)
        confidence = get_confidence_score(full_text)
        
        # 3. Participation Score (20%)
        participation = (data["total_time"] / total_time) * 100 if total_time > 0 else 0
        participation = round(participation, 2)
        
        # 4. Unique Points Score (25%)
        unique_count = get_unique_points(data["texts"])
        unique_score = min(unique_count * 10, 100)
        
        # Final Score
        final_score = (
            relevance * 0.40 +
            confidence * 0.15 +
            participation * 0.20 +
            unique_score * 0.25
        )
        
        results[name] = {
            "final_score": round(final_score, 2),
            "relevance": relevance,
            "confidence": confidence,
            "participation": round(participation, 2),
            "unique_points": unique_count
        }
    
    # ✅ NEW: Convert speaker keys to named segments
    named_segments = {}
    speaker_count = 0
    for speaker in speaker_segments:
        name = f"Debater {chr(65 + speaker_count)}"
        named_segments[name] = speaker_segments[speaker]
        speaker_count += 1

    return results, named_segments  # ✅ Now returns both

# =========================
# FORMAT OUTPUT
# =========================
def format_scores(topic, results, speaker_segments):
    output = f"📌 Topic: {topic}\n"
    output += "=" * 40 + "\n\n"
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]["final_score"], reverse=True)
    
    for rank, (name, scores) in enumerate(sorted_results, 1):
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "👤"
        output += f"{emoji} {name}: {scores['final_score']}/100\n"
        output += f"   ✅ Relevance:     {scores['relevance']}%\n"
        output += f"   💪 Confidence:    {scores['confidence']}%\n"
        output += f"   🎤 Participation: {scores['participation']}%\n"
        output += f"   💡 Unique Points: {scores['unique_points']}\n"
        output += f"   📝 Transcript:\n"
        for text in speaker_segments[name]["texts"]:
            output += f"      - {text}\n"
        output += "\n"
    
    return output