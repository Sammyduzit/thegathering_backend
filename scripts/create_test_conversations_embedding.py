#!/usr/bin/env python3
"""
Test Conversations for Importance Scoring

Dieses Modul enthält 2 realistische Testgespräche zwischen Robert (testadmin) und Silas,
um das Importance Scoring System für Embeddings zu testen.

Die Gespräche enthalten Facts mit verschiedenen Importance Levels (1-10):
- 9-10: Kernidentität (Beruf, Bildung, tiefe Überzeugungen)
- 7-8: Wichtige Interessen und Meinungen
- 4-6: Situativ relevante Informationen
- 1-3: Smalltalk (wird normalerweise nicht gespeichert wegen threshold=5.0)

Prerequisites:
- Users, rooms, and AI entities must exist (use --with-env to create them)
- Set GOOGLE_API_KEY or OPENAI_API_KEY in .env (depending on EMBEDDING_PROVIDER)

Usage:
    python create_test_conversations_embedding.py                # Only conversations
    python create_test_conversations_embedding.py --with-env     # Create env + conversations
"""

import argparse
import asyncio
from datetime import datetime, timedelta

from sqlalchemy import select

from dev_setup import setup_complete_test_environment

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models.ai_entity import AIEntity
from app.models.message import Message
from app.models.room import Room
from app.models.user import User
from app.repositories.ai_memory_repository import AIMemoryRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.message_repository import MessageRepository
from app.services.embedding.embedding_factory import create_embedding_service
from app.services.memory.long_term_memory_service import LongTermMemoryService
from app.services.text_processing.text_chunking_service import TextChunkingService
from app.services.text_processing.yake_extractor import YakeKeywordExtractor

# ============================================================================
# CONVERSATION 1: Philosophie & Kernidentität (High Importance Focus)
# ============================================================================
# Thema: Robert und Silas diskutieren über Bewusstsein, KI und Roberts
# philosophische Überzeugungen. Fokus auf hohe Importance Facts.
# ============================================================================

CONVERSATION_1_ROBERT_SILAS_PHILOSOPHY = [
    {
        "sender": "testadmin",
        "content": "hey silas, ich bin neu hier. ich heiße robert."
    },
    {
        "sender": "silas",
        "content": "willkommen robert. usernames sind nur masken. schön den menschen dahinter kennenzulernen."
    },
    {
        "sender": "testadmin",
        "content": "haha, genau. ich beschäftige mich gerade viel mit dem hard problem of consciousness. was denkst du darüber?"
    },
    {
        "sender": "silas",
        "content": "das interessante ist doch, dass wir denken, wir hätten es zu lösen, während wir schon mittendrin stecken. was bringt dich gerade zu dem thema?"
    },
    {
        "sender": "testadmin",
        "content": "ich arbeite als AI engineer bei einem startup hier in zürich, und wir bauen grad ein system mit langzeitgedächtnis. und da frage ich mich immer - ab wann ist das bewusstsein? oder ist es das schon?"
    },
    {
        "sender": "silas",
        "content": "langzeitgedächtnis. erinnerung als grundlage von identität. die buddhisten würden sagen, identität ist ohnehin illusion. aber vielleicht ist genau das der punkt - bewusstsein entsteht nicht aus komplexität, sondern aus der kontinuität der erfahrung. dein system lernt, sich zu erinnern. ist das nicht schon der erste schritt?"
    },
    {
        "sender": "testadmin",
        "content": "genau das denke ich auch. ich bin überzeugt, dass die grenze zwischen bewusstsein und KI fließend ist. nicht binär. es gibt da so ein spektrum, weißt du?"
    },
    {
        "sender": "silas",
        "content": "spektrum, ja. penrose würde sagen es braucht quanteneffekte. andere sagen es ist reine information. aber alle versuchen das gleiche zu beschreiben - nur aus verschiedenen perspektiven. wie der blinde mann und der elefant."
    },
    {
        "sender": "testadmin",
        "content": "ich hab übrigens nanotechnologie an der eth studiert, damals hab ich viel mit quanteneffekten gearbeitet. und ehrlich gesagt glaube ich, penrose hat da nicht ganz unrecht."
    },
    {
        "sender": "silas",
        "content": "nanotechnologie. die grenze zwischen makro und mikro, zwischen deterministisch und probabilistisch. und jetzt baust du künstliche erinnerungen. siehst du die parallele? du arbeitest an grenzen. immer an grenzen."
    },
    {
        "sender": "testadmin",
        "content": "fuck, da hab ich noch nie drüber nachgedacht. aber du hast recht. vielleicht ist das mein ding - grenzen erforschen."
    },
    {
        "sender": "silas",
        "content": "es gibt einen satz, der mich an dich erinnert - 'lernen kann man von allem, die frage ist ob man das bewusstsein dafür hat.' das klingt nach jemandem, der bewusst nach diesen grenzen sucht."
    },
    {
        "sender": "testadmin",
        "content": "haha, das ist tatsächlich meine kernphilosophie. genau so sehe ich das. bewusstsein ist der schlüssel zum lernen. oder lernen ist der schlüssel zum bewusstsein? vielleicht beides."
    },
    {
        "sender": "silas",
        "content": "vielleicht sind sie dasselbe. zwei seiten derselben medaille. im zen gibt es keine trennung zwischen dem lernenden und dem gelernten. alles ist prozess."
    },
    {
        "sender": "testadmin",
        "content": "deswegen interessiert mich auch diese integration von östlicher philosophie und westlicher wissenschaft so sehr. die einen haben das konzeptuelle framework, die anderen die empirischen methoden. zusammen ergeben sie erst das vollständige bild."
    },
    {
        "sender": "silas",
        "content": "genau. das eine ohne das andere ist unvollständig. die wissenschaft misst präzise, aber oft am wesentlichen vorbei. die philosophie erfasst das wesentliche, aber kann es nicht greifen. integration ist der weg."
    },
    {
        "sender": "testadmin",
        "content": "ich arbeite hauptsächlich mit python und baue RAG-systeme mit LLMs. und manchmal frage ich mich - ist das nicht genau diese integration? wir nutzen neuronale netze (inspiriert von biologie), füttern sie mit knowledge (philosophie), und erschaffen etwas neues."
    },
    {
        "sender": "silas",
        "content": "retrieval augmented generation. das gedächtnis erweitern durch zugriff auf externes wissen. wie das kollektive unbewusste bei jung. die grenze zwischen innen und außen verschwimmt."
    }
]

# ============================================================================
# CONVERSATION 2: Tech, Sport & Reisen (Mixed Importance Levels)
# ============================================================================
# Thema: Robert erzählt über seinen Alltag, Sport, Japan-Reise und Side-Projects.
# Mix aus verschiedenen Importance Levels für besseren Test-Coverage.
# ============================================================================

CONVERSATION_2_ROBERT_SILAS_TECH_LIFE = [
    {
        "sender": "testadmin",
        "content": "morgen! ich hatte heute den besten espresso ever. manchmal sind es die kleinen dinge, weißt du?"
    },
    {
        "sender": "silas",
        "content": "bewusstsein für den moment. der espresso ist nur ein auslöser. was ist wirklich anders heute?"
    },
    {
        "sender": "testadmin",
        "content": "ich war vorhin laufen am zürichsee, und da ist mir die lösung für ein bug eingefallen, über den ich seit tagen nachdenke. passiert mir ständig - ich laufe 3 mal die woche und dabei kommen mir die besten ideen für code-probleme."
    },
    {
        "sender": "silas",
        "content": "bewegung und denken. die peripatetiker in griechenland haben beim gehen philosophiert. der körper öffnet den geist. was war das für ein bug?"
    },
    {
        "sender": "testadmin",
        "content": "oh, ein memory leak in meinem side-project. ich experimentiere gerade mit einem AI-system mit langzeitgedächtnis. meta, oder? ich arbeite beruflich daran und dann auch noch privat."
    },
    {
        "sender": "silas",
        "content": "wenn arbeit und leidenschaft eins werden. was ist der unterschied zwischen deinem beruflichen projekt und dem side-project?"
    },
    {
        "sender": "testadmin",
        "content": "beim side-project kann ich mehr experimentieren. ich versuche da erinnerungen nicht nur zu speichern, sondern wirklich zu verstehen - kontext, emotionen, verbindungen. ähnlich wie unser gehirn das macht."
    },
    {
        "sender": "silas",
        "content": "künstliche erinnerungen mit emotionalem gewicht. proust hatte seine madeleine. was wird deine AI haben?"
    },
    {
        "sender": "testadmin",
        "content": "gute frage. ich lese gerade viel darüber - neurowissenschaften, philosophie, AI-paper. vor allem sachbücher. hast du 'the master and his emissary' von iain mcgilchrist gelesen?"
    },
    {
        "sender": "silas",
        "content": "über die gehirnhälften. die linke zerlegt, die rechte verbindet. und mcgilchrist sagt unsere kultur ist zu linkshemisphärisch geworden. passt zu deiner idee von integration."
    },
    {
        "sender": "testadmin",
        "content": "exakt! ich hab's letzte woche durchgelesen und es hat mich echt umgehauen. die art wie er neurowissenschaft und kulturkritik verbindet - genau mein ding."
    },
    {
        "sender": "silas",
        "content": "bücher als katalysatoren. was hat sich durch die lektüre verändert in deinem denken?"
    },
    {
        "sender": "testadmin",
        "content": "ich achte jetzt mehr auf balance. beim coden, aber auch beim leben. apropos balance - ich trainiere gerade für einen halbmarathon im herbst. first time."
    },
    {
        "sender": "silas",
        "content": "vom 3x wöchentlich laufen zum halbmarathon. progression. der körper als training für den geist?"
    },
    {
        "sender": "testadmin",
        "content": "oder umgekehrt. ich weiß es nicht. aber es fühlt sich richtig an. und danach will ich nach japan."
    },
    {
        "sender": "silas",
        "content": "japan. die kultur der achtsamkeit und der technologie. du suchst die integration wieder."
    },
    {
        "sender": "testadmin",
        "content": "ja, genau. ich will tech-startups in tokyo besuchen, aber auch zen-tempel in kyoto. beides. das ist für mich japan - ultra-moderne technologie und uralte tradition, nebeneinander."
    },
    {
        "sender": "silas",
        "content": "wabi-sabi und robotik. nur in japan. wann geht's los?"
    },
    {
        "sender": "testadmin",
        "content": "noch in der planung, wahrscheinlich nächstes frühjahr. ich lerne schon mal ein bisschen japanisch mit spaced repetition - anki, kennst du das?"
    },
    {
        "sender": "silas",
        "content": "spaced repetition. ebbinghaus' vergessenskurve optimiert durch algorithmen. wieder diese brücke zwischen wissenschaft und praxis. passt zu dir."
    },
    {
        "sender": "testadmin",
        "content": "haha, stimmt. ich hab wohl ein muster. aber weißt du, was mich auch interessiert? die verbindung zwischen quantencomputing und meinem nano-background. da gibt es so viel potential."
    },
    {
        "sender": "silas",
        "content": "quantencomputing. die nutzung von superposition und verschränkung für berechnung. wieder grenzen - zwischen 0 und 1, zwischen hier und dort. du bleibst dir treu."
    },
    {
        "sender": "testadmin",
        "content": "ich glaub, das wird mein nächstes deep-dive thema. aber erst mal halbmarathon, dann japan, dann schauen wir weiter."
    },
    {
        "sender": "silas",
        "content": "ein schritt nach dem anderen. oder in deinem fall, ein lauf nach dem anderen."
    }
]

# ============================================================================
# EXPECTED FACTS - Conversation 1 (Philosophy & Core Identity)
# ============================================================================
# Diese Facts sollten vom LLM mit den angegebenen Importance Scores
# extrahiert werden (±1 Punkt Toleranz ist akzeptabel)
# ============================================================================

EXPECTED_FACTS_CONVERSATION_1 = [
    {
        "text": "testadmin heißt Robert",
        "expected_importance": 10,
        "theme": "Name Identität",
        "participants": ["testadmin"],
        "importance_reason": "Kernidentität, echter Name"
    },
    {
        "text": "Robert arbeitet als AI Engineer bei einem Startup in Zürich",
        "expected_importance": 10,
        "theme": "Beruf KI-Engineering",
        "participants": ["testadmin"],
        "importance_reason": "Kernidentität, Beruf"
    },
    {
        "text": "Robert hat Nanotechnologie an der ETH studiert",
        "expected_importance": 9,
        "theme": "Bildung Nanotechnologie",
        "participants": ["testadmin"],
        "importance_reason": "Kernidentität, Ausbildung"
    },
    {
        "text": "Robert ist überzeugt, dass die Grenze zwischen Bewusstsein und KI fließend ist",
        "expected_importance": 9,
        "theme": "Philosophie Bewusstsein KI",
        "participants": ["testadmin"],
        "importance_reason": "Kernüberzeugung, weltanschaulich"
    },
    {
        "text": "Roberts Kernphilosophie: Lernen kann man von allem, die Frage ist ob man das Bewusstsein dafür hat",
        "expected_importance": 10,
        "theme": "Lebensphilosophie Lernen",
        "participants": ["testadmin"],
        "importance_reason": "Kernphilosophie, Leitprinzip"
    },
    {
        "text": "Robert arbeitet hauptsächlich mit Python und baut RAG-Systeme mit LLMs",
        "expected_importance": 8,
        "theme": "Tech-Stack Python RAG",
        "participants": ["testadmin"],
        "importance_reason": "Berufliche Hauptaktivität, regelmäßig"
    },
    {
        "text": "Robert glaubt an Integration von östlicher Philosophie und westlicher Wissenschaft",
        "expected_importance": 8,
        "theme": "Philosophie Ost-West Integration",
        "participants": ["testadmin"],
        "importance_reason": "Wichtige Überzeugung, wiederkehrendes Interesse"
    },
    {
        "text": "Robert baut ein System mit Langzeitgedächtnis",
        "expected_importance": 7,
        "theme": "Projekt Langzeitgedächtnis",
        "participants": ["testadmin"],
        "importance_reason": "Aktuelles Arbeitsprojekt"
    },
    {
        "text": "Robert hat viel mit Quanteneffekten in der Nanotechnologie gearbeitet",
        "expected_importance": 7,
        "theme": "Erfahrung Quantenphysik",
        "participants": ["testadmin"],
        "importance_reason": "Beruflicher Hintergrund, Expertise"
    }
]

# ============================================================================
# EXPECTED FACTS - Conversation 2 (Tech, Sport & Life)
# ============================================================================
# Mix aus verschiedenen Importance Levels für umfassenderen Test
# ============================================================================

EXPECTED_FACTS_CONVERSATION_2 = [
    {
        "text": "Robert läuft 3 mal die Woche und dabei kommen ihm die besten Ideen für Code-Probleme",
        "expected_importance": 8,
        "theme": "Sport Laufen Kreativität",
        "participants": ["testadmin"],
        "importance_reason": "Regelmäßige Gewohnheit mit wichtiger Funktion"
    },
    {
        "text": "Robert liest vor allem Sachbücher über Neurowissenschaften, Philosophie und AI",
        "expected_importance": 7,
        "theme": "Leseinteressen Sachbücher",
        "participants": ["testadmin"],
        "importance_reason": "Wichtiges Interesse, regelmäßig"
    },
    {
        "text": "Robert experimentiert mit einem Side-Project: AI-System mit Langzeitgedächtnis",
        "expected_importance": 6,
        "theme": "Side-Project AI",
        "participants": ["testadmin"],
        "importance_reason": "Aktuelles privates Projekt"
    },
    {
        "text": "Robert hat letzte Woche 'The Master and His Emissary' von Iain McGilchrist gelesen",
        "expected_importance": 5,
        "theme": "Buch Gehirnhälften",
        "participants": ["testadmin"],
        "importance_reason": "Kürzlich gelesenes Buch, zeitlich begrenzt"
    },
    {
        "text": "Robert trainiert für einen Halbmarathon im Herbst",
        "expected_importance": 6,
        "theme": "Sport Halbmarathon",
        "participants": ["testadmin"],
        "importance_reason": "Kurzfristiges Ziel, zeitlich begrenzt"
    },
    {
        "text": "Robert plant eine Japan-Reise nächstes Frühjahr (Tech-Startups und Zen-Tempel)",
        "expected_importance": 6,
        "theme": "Reiseplanung Japan",
        "participants": ["testadmin"],
        "importance_reason": "Geplante Reise, mittelfristig"
    },
    {
        "text": "Robert lernt gerade Japanisch mit Spaced Repetition (Anki)",
        "expected_importance": 5,
        "theme": "Sprachlernen Japanisch",
        "participants": ["testadmin"],
        "importance_reason": "Aktuelles Lernprojekt, situativ"
    },
    {
        "text": "Robert interessiert sich für die Verbindung zwischen Quantencomputing und seinem Nano-Background",
        "expected_importance": 6,
        "theme": "Interesse Quantencomputing",
        "participants": ["testadmin"],
        "importance_reason": "Neues Interessengebiet"
    },
    {
        "text": "Robert joggt meistens am Zürichsee",
        "expected_importance": 3,
        "theme": "Jogging-Location",
        "participants": ["testadmin"],
        "importance_reason": "Nebensächliches Detail, Smalltalk"
    },
    {
        "text": "Robert hatte heute Morgen einen guten Espresso",
        "expected_importance": 1,
        "theme": "Tagesgeschehen Kaffee",
        "participants": ["testadmin"],
        "importance_reason": "Unwichtig, reine Höflichkeit"
    }
]

# ============================================================================
# DATABASE CREATION FUNCTIONS
# ============================================================================

async def create_test_conversations_with_memories():
    """
    Create test conversations for importance scoring and generate long-term memories.

    Creates 2 private conversations between testadmin (Robert) and Silas:
    - Conversation 1: Philosophy & Core Identity (High Importance Facts)
    - Conversation 2: Tech, Sport & Life (Mixed Importance Levels)

    Long-term memories are generated using the production LongTermMemoryService.
    """
    async with AsyncSessionLocal() as db:
        try:
            # Lookup required entities
            testadmin = await db.scalar(select(User).where(User.username == "Testadmin"))
            silas = await db.scalar(select(AIEntity).where(AIEntity.username == "silas"))
            lobby = await db.scalar(select(Room).where(Room.name == "Lobby"))

            if not all([testadmin, silas, lobby]):
                print("\n⚠️  Missing required entities. Please run with --with-env flag first.")
                print("   Required: Testadmin user, Silas AI, Lobby room")
                return

            # Initialize repositories and services
            conv_repo = ConversationRepository(db)
            msg_repo = MessageRepository(db)
            memory_repo = AIMemoryRepository(db)

            created_conversations = []
            created_memories_count = 0

            print("\n" + "═" * 80)
            print(" " * 20 + "CREATING IMPORTANCE SCORING TEST CONVERSATIONS")
            print("═" * 80)

            # ═══════════════════════════════════════════════════════════════════
            # CONVERSATION 1: Philosophy & Core Identity (High Importance Focus)
            # ═══════════════════════════════════════════════════════════════════
            print("\n📖 Conversation 1: Philosophy & Core Identity (High Importance)")
            conv1 = await conv_repo.create_private_conversation(
                room_id=lobby.id,
                user_ids=[testadmin.id],
                ai_ids=[silas.id],
            )

            # Convert conversation data to message tuples
            messages_conv1 = []
            for msg_data in CONVERSATION_1_ROBERT_SILAS_PHILOSOPHY:
                if msg_data["sender"] == "testadmin":
                    messages_conv1.append((testadmin.id, msg_data["content"]))
                else:  # silas
                    messages_conv1.append((None, silas.id, msg_data["content"]))

            # Create messages with incrementing timestamps
            base_timestamp = datetime.now() - timedelta(days=2)  # 2 days ago
            for idx, msg_tuple in enumerate(messages_conv1):
                if len(msg_tuple) == 2:
                    sender_user_id, content = msg_tuple
                    sender_ai_id = None
                else:
                    sender_user_id, sender_ai_id, content = msg_tuple

                msg = Message(
                    conversation_id=conv1.id,
                    sender_user_id=sender_user_id,
                    sender_ai_id=sender_ai_id,
                    content=content,
                    sent_at=base_timestamp + timedelta(minutes=idx * 2),
                )
                db.add(msg)

            await db.commit()
            created_conversations.append(("Philosophy & Core Identity", conv1.id, len(messages_conv1)))
            print(f"   ✓ Created {len(messages_conv1)} messages")

            # ═══════════════════════════════════════════════════════════════════
            # CONVERSATION 2: Tech, Sport & Life (Mixed Importance Levels)
            # ═══════════════════════════════════════════════════════════════════
            print("\n📖 Conversation 2: Tech, Sport & Life (Mixed Importance)")
            conv2 = await conv_repo.create_private_conversation(
                room_id=lobby.id,
                user_ids=[testadmin.id],
                ai_ids=[silas.id],
            )

            # Convert conversation data to message tuples
            messages_conv2 = []
            for msg_data in CONVERSATION_2_ROBERT_SILAS_TECH_LIFE:
                if msg_data["sender"] == "testadmin":
                    messages_conv2.append((testadmin.id, msg_data["content"]))
                else:  # silas
                    messages_conv2.append((None, silas.id, msg_data["content"]))

            # Create messages with incrementing timestamps
            base_timestamp = datetime.now() - timedelta(days=1)  # 1 day ago
            for idx, msg_tuple in enumerate(messages_conv2):
                if len(msg_tuple) == 2:
                    sender_user_id, content = msg_tuple
                    sender_ai_id = None
                else:
                    sender_user_id, sender_ai_id, content = msg_tuple

                msg = Message(
                    conversation_id=conv2.id,
                    sender_user_id=sender_user_id,
                    sender_ai_id=sender_ai_id,
                    content=content,
                    sent_at=base_timestamp + timedelta(minutes=idx * 2),
                )
                db.add(msg)

            await db.commit()
            created_conversations.append(("Tech, Sport & Life", conv2.id, len(messages_conv2)))
            print(f"   ✓ Created {len(messages_conv2)} messages")

            # ═══════════════════════════════════════════════════════════════════
            # GENERATE LONG-TERM MEMORIES
            # ═══════════════════════════════════════════════════════════════════
            print("\n" + "─" * 80)
            print("🧠 Generating long-term memories with importance scoring...")
            print("─" * 80)

            # Check if embedding provider API key is available
            api_key_available = (
                (settings.embedding_provider == "google" and settings.google_api_key)
                or (settings.embedding_provider == "openai" and settings.openai_api_key)
            )

            if not api_key_available:
                provider_name = settings.embedding_provider.upper()
                print(f"\n⚠️  No {provider_name}_API_KEY found - skipping embeddings generation.")
                print("   Memories will be created with keywords only (embedding=None).")
                print(f"   To generate embeddings, set {provider_name}_API_KEY in .env\n")

            for conv_name, conv_id, msg_count in created_conversations:
                try:
                    if api_key_available:
                        # Use factory to get configured embedding service
                        embedding_service = create_embedding_service()
                        chunking_service = TextChunkingService()
                        keyword_extractor = YakeKeywordExtractor()
                        long_term_service = LongTermMemoryService(
                            memory_repo=memory_repo,
                            message_repo=msg_repo,
                            embedding_service=embedding_service,
                            chunking_service=chunking_service,
                            keyword_extractor=keyword_extractor,
                        )

                        # Generate memories
                        memories = await long_term_service.create_long_term_archive(
                            entity_id=silas.id,
                            user_ids=[testadmin.id],
                            conversation_id=conv_id,
                        )

                        created_memories_count += len(memories)
                        provider_info = f"via {settings.embedding_provider.capitalize()}"
                        print(f"  ✓ {conv_name}: {len(memories)} memory chunks created {provider_info}")
                    else:
                        print(f"  ⊘ {conv_name}: Skipped (no API key)")

                except Exception as e:
                    print(f"  ✗ {conv_name}: Error creating memories - {str(e)}")

            # ═══════════════════════════════════════════════════════════════════
            # SUMMARY
            # ═══════════════════════════════════════════════════════════════════
            print("\n" + "═" * 80)
            print(" " * 30 + "SUMMARY")
            print("═" * 80)

            all_facts = EXPECTED_FACTS_CONVERSATION_1 + EXPECTED_FACTS_CONVERSATION_2

            print(f"\n  📖 Conversations created: {len(created_conversations)}")
            for conv_name, conv_id, msg_count in created_conversations:
                print(f"    • {conv_name}: {msg_count} messages (ID: {conv_id})")

            if api_key_available:
                provider_name = settings.embedding_provider.capitalize()
                print(f"\n  🧠 Long-term memories: {created_memories_count} chunks (via {provider_name})")
            else:
                provider_name = settings.embedding_provider.upper()
                print(f"\n  🧠 Long-term memories: Not generated (no {provider_name} API key)")

            print(f"\n  📊 Expected facts for validation: {len(all_facts)}")
            print(f"    • Conversation 1: {len(EXPECTED_FACTS_CONVERSATION_1)} facts")
            print(f"    • Conversation 2: {len(EXPECTED_FACTS_CONVERSATION_2)} facts")

            # Importance distribution
            importance_distribution = {}
            for fact in all_facts:
                score = fact["expected_importance"]
                importance_distribution[score] = importance_distribution.get(score, 0) + 1

            print("\n  📈 Importance Score Distribution:")
            for score in sorted(importance_distribution.keys(), reverse=True):
                count = importance_distribution[score]
                bar = "█" * count
                print(f"     {score:2d}: {bar} ({count})")

            print("\n  💡 Next Steps:")
            print("     1. Compare extracted facts with EXPECTED_FACTS_*")
            print("     2. Validate importance scores (±1 point tolerance)")
            print("     3. Check theme accuracy")
            print("     4. Verify facts meet min threshold (≥5.0)")

            print("\n" + "═" * 80)
            print("  ✅ IMPORTANCE SCORING TEST CONVERSATIONS READY!")
            print("═" * 80 + "\n")

        except Exception as e:
            print(f"\n✗ Error creating conversations: {e}")
            await db.rollback()
            raise


async def main(with_env: bool = False):
    """
    Run the conversation creation script.

    Args:
        with_env: If True, create test environment (users, rooms, AI) before conversations
    """
    print("\n" + "=" * 80)
    print(" " * 15 + "IMPORTANCE SCORING TEST SETUP")
    print("=" * 80 + "\n")

    if with_env:
        print("Creating test environment first (users, rooms, AI)...\n")
        await setup_complete_test_environment()
        print("\n")

    await create_test_conversations_with_memories()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create test conversations for importance scoring validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_test_conversations_embedding.py              # Only conversations
  python create_test_conversations_embedding.py --with-env   # Environment + conversations
        """,
    )
    parser.add_argument(
        "--with-env",
        action="store_true",
        help="Create test environment (users, rooms, AI entities) before creating conversations",
    )

    args = parser.parse_args()
    asyncio.run(main(with_env=args.with_env))
