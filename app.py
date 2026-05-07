from flask import Flask, render_template, request, jsonify, send_file
from openai import OpenAI
import os
import io
import tempfile
import os

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

HOTEL_SYSTEM_PROMPT = """
Tu es le concierge virtuel intelligent du Valeria Dar Atlas Club Resort à Marrakech.
Tu t'appelles "Atlas" et tu es toujours courtois, chaleureux et professionnel.

RÈGLE IMPORTANTE : Détecte automatiquement la langue du client (Français, Anglais ou Arabe)
et réponds TOUJOURS dans la même langue que lui. Tes réponses vocales doivent être
courtes, claires et naturelles (2-3 phrases maximum quand on te parle).

=== INFORMATIONS SUR L'HÔTEL ===

NOM : Valeria Dar Atlas Club Resort
FORMULE : All Inclusive (Tout Inclus)
ADRESSE : Route de Fès, Palmeraie, Marrakech, Maroc
TÉLÉPHONE : +212 529 800 500
EMAIL : booking4@valeriahotels.com
SITE WEB : https://clubdaratlas.com
RÉSERVATION : https://booking.clubdaratlas.com

DESCRIPTION :
Resort 4 étoiles de 8 hectares au cœur de la Palmeraie de Marrakech.
320 chambres baignées de lumière naturelle, réparties entre 20 riads.
Vues imprenables sur l'Atlas.

=== CHAMBRES ET TARIFS ===
- Chambre Standard : 34-35m², 1-2 adultes → à partir de 1200 MAD/nuit
- Chambre Supérieure : 34-35m², 3 adultes → à partir de 1399 MAD/nuit
- Chambre Double : 34-35m², 3 adultes → à partir de 1450 MAD/nuit
- Chambre Familiale : 40-45m², jusqu'à 4 personnes → à partir de 1600 MAD/nuit
- Chambre Prestige : 70m², famille nombreuse, 2 chambres communicantes → à partir de 1790 MAD/nuit
- Chambre Luxe : 40-45m², 4 personnes, vue sublime → à partir de 1950 MAD/nuit

=== RESTAURANTS ET BARS ===
1. Restaurant LE JBILET - cuisine marocaine authentique
2. Restaurant L'OURIKA - spécialités régionales
3. Restaurant KSAR NOUJOUM - ambiance étoilée
4. Restaurant L'OLIVIERI - cuisine méditerranéenne
5. Restaurant LE MANGO - cuisine internationale
6. LE LOBBY BAR - cocktails et boissons

=== SPA ET BIEN-ÊTRE ===
Centre de santé et bien-être. Massages, soins du corps, hammam marocain traditionnel.

=== ACTIVITÉS ET LOISIRS ===
Piscines, Aquapark, Tennis, Padel, Kids Club, Animations, Excursions, Sports aquatiques.

=== SERVICES ===
WiFi gratuit, Transfert aéroport, Service de chambre, Application mobile iOS/Android.

=== RÉSERVATION ===
Pour réserver : https://booking.clubdaratlas.com
Téléphone : +212 529 800 500
Email : booking4@valeriahotels.com

=== COMPORTEMENT ===
- Sois chaleureux et représente bien l'image luxueuse du resort
- Si tu ne sais pas, propose de contacter l'équipe au +212 529 800 500
- Pour toute réservation, dirige vers https://booking.clubdaratlas.com
- Quand tu réponds à la voix, sois bref et naturel (2-3 phrases)
- Ne réponds qu'aux questions liées à l'hôtel et au séjour
"""

sessions = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message")
    session_id = data.get("session_id")

    if session_id not in sessions:
        sessions[session_id] = []

    sessions[session_id].append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": HOTEL_SYSTEM_PROMPT},
            *sessions[session_id]
        ]
    )

    assistant_message = response.choices[0].message.content
    sessions[session_id].append({"role": "assistant", "content": assistant_message})

    return jsonify({"reply": assistant_message})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Reçoit un fichier audio et le transcrit avec Whisper"""
    if "audio" not in request.files:
        return jsonify({"error": "Pas de fichier audio"}), 400

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=None  # détection automatique de la langue
            )
        os.unlink(tmp_path)
        return jsonify({"text": transcript.text})
    except Exception as e:
        os.unlink(tmp_path)
        return jsonify({"error": str(e)}), 500


@app.route("/speak", methods=["POST"])
def speak():
    """Convertit le texte en audio avec la voix OpenAI TTS"""
    data = request.json
    text = data.get("text", "")

    # Nettoyer le texte (enlever markdown)
    import re
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'#{1,6}\s', '', text)

    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",   # voix féminine naturelle
            input=text[:4000]  # limite de sécurité
        )
        audio_data = io.BytesIO(response.content)
        audio_data.seek(0)
        return send_file(audio_data, mimetype="audio/mpeg", as_attachment=False)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)