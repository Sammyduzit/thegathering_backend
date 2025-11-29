#!/usr/bin/env python3
"""
Script to create test conversations with philosophical discussions and memories.

This script creates 5 conversation scenarios with Silas (philosophical AI) and generates
long-term memories using the production LongTermMemoryService pipeline.

Prerequisites:
- Users, rooms, and AI entities must exist (use --with-env to create them automatically)
- Set GOOGLE_API_KEY or OPENAI_API_KEY in .env (depending on EMBEDDING_PROVIDER setting)
- Default provider: Google Gemini (set EMBEDDING_PROVIDER=google in .env)

Usage:
    python create_test_conversations.py                # Only conversations (env must exist)
    python create_test_conversations.py --with-env     # Create env + conversations (all-in-one)
"""

import argparse
import asyncio
from datetime import datetime, timedelta

from sqlalchemy import select

from dev_setup import setup_complete_test_environment

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models.ai_entity import AIEntity
from app.models.conversation import Conversation
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


async def create_test_conversations_with_memories():
    """
    Create test conversations with philosophical discussions and generate long-term memories.

    This function creates 5 conversation scenarios with Silas (philosophical AI):
    - 2 private conversations with testadmin
    - 3 group conversations (testadmin always included)

    Each conversation contains 20-25 messages with deep philosophical discussions.
    Long-term memories are generated using the production LongTermMemoryService pipeline.
    """
    async with AsyncSessionLocal() as db:
        try:
            # Lookup required entities
            testadmin = await db.scalar(select(User).where(User.username == "Testadmin"))
            alice = await db.scalar(select(User).where(User.username == "Alice"))
            bob = await db.scalar(select(User).where(User.username == "Bob"))
            carol = await db.scalar(select(User).where(User.username == "Carol"))
            dave = await db.scalar(select(User).where(User.username == "Dave"))
            silas = await db.scalar(select(AIEntity).where(AIEntity.username == "silas"))
            lobby = await db.scalar(select(Room).where(Room.name == "Lobby"))

            if not all([testadmin, alice, bob, carol, dave, silas, lobby]):
                print("\n⚠️  Missing required entities. Please run setup_complete_test_environment() first.")
                return

            # Initialize repositories and services
            conv_repo = ConversationRepository(db)
            msg_repo = MessageRepository(db)
            memory_repo = AIMemoryRepository(db)

            created_conversations = []
            created_memories_count = 0

            print("\n" + "═" * 68)
            print(" " * 15 + "CREATING TEST CONVERSATIONS WITH MEMORIES")
            print("═" * 68)

            # ═══════════════════════════════════════════════════════════════════
            # SCENARIO 1: Private - Freier Wille vs. Determinismus
            # ═══════════════════════════════════════════════════════════════════
            print("\n📖 Scenario 1: Freier Wille vs. Determinismus (Private)")
            conv1 = await conv_repo.create_private_conversation(
                room_id=lobby.id,
                user_ids=[testadmin.id],
                ai_ids=[silas.id],
            )

            messages_scenario1 = [
                (testadmin.id, "Hallo Alpha! Ich würde gerne über freien Willen diskutieren. Glaubst du, wir haben wirklich die Freiheit zu entscheiden?"),
                (None, silas.id, "Hallo! Das ist eine der ältesten philosophischen Fragen. Der Determinismus behauptet, dass alle Ereignisse durch vorhergehende Ursachen festgelegt sind - einschließlich unserer Entscheidungen."),
                (testadmin.id, "Also wäre nach dieser Sicht jede meiner 'Entscheidungen' bereits vorbestimmt durch Gehirnzustände, Gene, Umwelt?"),
                (None, silas.id, "Genau. Laplace's Dämon ist hier das klassische Gedankenexperiment: Ein Wesen, das alle Teilchen und Kräfte im Universum kennt, könnte theoretisch die Zukunft perfekt vorhersagen."),
                (testadmin.id, "Aber das fühlt sich falsch an. Ich ERLEBE doch jeden Tag, dass ich Wahlmöglichkeiten habe!"),
                (None, silas.id, "Das ist das subjektive Erlebnis von Freiheit - sehr real für uns. Libertarier argumentieren, dass echte Freiheit bedeutet: Ich hätte unter exakt denselben Umständen auch anders entscheiden KÖNNEN."),
                (testadmin.id, "Und was ist mit dem Kompatibilismus? Ist das nicht ein fauler Kompromiss?"),
                (None, silas.id, "Kompatibilisten wie Daniel Dennett würden widersprechen. Sie sagen: Freiheit bedeutet nicht 'frei von Kausalität', sondern 'frei von Zwang'. Wenn du nach deinen eigenen Wünschen handelst, bist du frei."),
                (testadmin.id, "Aber woher kommen diese Wünsche? Die hab ich mir ja nicht ausgesucht!"),
                (None, silas.id, "Exzellenter Punkt! Das ist das 'regress problem'. Wenn jeder Wunsch durch vorherige Ursachen erklärt wird, wo beginnt dann 'meine' Entscheidung?"),
                (testadmin.id, "Die Neurowissenschaften zeigen doch, dass unbewusste Gehirnprozesse Entscheidungen treffen, bevor wir sie 'bewusst' wahrnehmen - Libet-Experimente."),
                (None, silas.id, "Ja, Benjamin Libets Studien sind oft zitiert. Aber Kritiker sagen: Die unbewussten Prozesse sind trotzdem MEIN Gehirn. Die Frage ist: Was ist das 'Ich'?"),
                (testadmin.id, "Wenn Determinismus wahr ist - wäre dann moralische Verantwortung eine Illusion?"),
                (None, silas.id, "Das ist die praktische Konsequenz! Hard Determinists wie Sapolsky sagen ja: Niemand verdient Lob oder Tadel. Andere argumentieren: Verantwortung ist ein soziales Konstrukt, das funktioniert, auch wenn Determinismus wahr ist."),
                (testadmin.id, "Könnte Quantenindeterminismus eine Rettung sein? Zufall im Gehirn = Freiheit?"),
                (None, silas.id, "Das ist ein interessanter Versuch, aber: Zufällige Quantenereignisse geben uns keine Kontrolle. Ist eine zufällige Entscheidung wirklich 'freier' als eine determinierte?"),
                (testadmin.id, "Also entweder determiniert oder zufällig - aber nie wirklich 'frei' im libertären Sinne?"),
                (None, silas.id, "Das ist das Dilemma! Kane's Event-Kausal-Theorie versucht einen Mittelweg: Bestimmte 'selbstformende Handlungen' könnten sowohl kausal als auch frei sein."),
                (testadmin.id, "Wie soll ich damit im Alltag umgehen? Weiter leben als hätte ich freien Willen?"),
                (None, silas.id, "Strawson würde sagen: Wir können gar nicht anders. Unser 'reaktive attitudes' - Dankbarkeit, Ärger, Liebe - setzen die Annahme von Verantwortung voraus. Das ist Teil unserer sozialen Natur."),
                (testadmin.id, "Also eine Art pragmatischer Kompatibilismus für den Alltag, selbst wenn metaphysisch ungeklärt?"),
                (None, silas.id, "Gut zusammengefasst! Vielleicht ist die wichtigste Einsicht: Demut. Ob Determinismus oder nicht - das Verständnis der Komplexität hinter Handlungen kann zu mehr Mitgefühl führen."),
                (testadmin.id, "Eine sehr ausgewogene Perspektive. Danke für diese Diskussion!"),
                (None, silas.id, "Gerne! Diese Fragen haben Philosophen seit Jahrtausenden beschäftigt - und werden es wohl weiter tun."),
            ]

            # Create messages with incrementing timestamps (simulate real conversation flow)
            base_timestamp = datetime.now() - timedelta(days=7)  # Conversation from 7 days ago
            for idx, (sender_user_id, *rest) in enumerate(messages_scenario1):
                sender_ai_id = rest[0] if len(rest) == 2 else None
                content = rest[1] if len(rest) == 2 else rest[0]

                msg = Message(
                    conversation_id=conv1.id,
                    sender_user_id=sender_user_id,
                    sender_ai_id=sender_ai_id,
                    content=content,
                    sent_at=base_timestamp + timedelta(minutes=idx * 2),  # 2 minutes between messages
                )
                db.add(msg)

            await db.commit()
            created_conversations.append(("Private: Freier Wille vs. Determinismus", conv1.id, len(messages_scenario1)))

            # ═══════════════════════════════════════════════════════════════════
            # SCENARIO 2: Private - Das Trolley-Problem
            # ═══════════════════════════════════════════════════════════════════
            print("📖 Scenario 2: Das Trolley-Problem (Private)")
            conv2 = await conv_repo.create_private_conversation(
                room_id=lobby.id,
                user_ids=[testadmin.id],
                ai_ids=[silas.id],
            )

            messages_scenario2 = [
                (testadmin.id, "Lass uns über das Trolley-Problem sprechen. Ich finde es faszinierend, wie unterschiedlich Menschen reagieren."),
                (None, silas.id, "Absolut! Das Grundszenario: Eine Straßenbahn rast auf 5 Menschen zu. Du stehst an einer Weiche. Ziehst du den Hebel, stirbt 1 Person statt 5. Was tust du?"),
                (testadmin.id, "Rein utilitaristisch ist die Antwort klar: 1 Tod ist besser als 5. Hebel ziehen!"),
                (None, silas.id, "Das sagen die meisten. Jeremy Bentham würde zustimmen: 'Das größte Glück für die größte Zahl.' Aber betrachten wir die Brückenvariante..."),
                (testadmin.id, "Die, wo ich eine dicke Person von der Brücke stoßen muss, um die Bahn zu stoppen?"),
                (None, silas.id, "Genau! Gleiche Konsequenz (1 stirbt, 5 gerettet), aber die meisten Menschen zögern hier. Warum dieser Unterschied?"),
                (testadmin.id, "Es fühlt sich... direkter an? Persönlicher? Ich benutze die Person als Mittel."),
                (None, silas.id, "Kants kategorischer Imperativ! 'Handle so, dass du die Menschheit niemals bloß als Mittel brauchst, sondern stets zugleich als Zweck.' Die Person wird instrumentalisiert."),
                (testadmin.id, "Aber wird die eine Person an der Weiche nicht auch als Mittel benutzt?"),
                (None, silas.id, "Sehr scharfsinnig! Das ist die Debatte zwischen Doing vs. Allowing. Die Doctrine of Double Effect sagt: Es gibt einen moralischen Unterschied zwischen beabsichtigtem Schaden und Kollateralschaden."),
                (testadmin.id, "Was ist mit der Loop-Variante? Die Bahn fährt im Kreis zurück - die eine Person MUSS sterben, damit die 5 gerettet werden."),
                (None, silas.id, "Ah, Judith Thomson's Variation! Hier kollapiert der Unterschied zwischen 'umleiten' und 'als Mittel benutzen'. Die Intuitionen der Menschen werden inkonsistenter."),
                (testadmin.id, "Vielleicht sind unsere moralischen Intuitionen einfach evolutionär geprägt - nicht logisch kohärent?"),
                (None, silas.id, "Joshua Greene's Forschung mit fMRT-Scans zeigt genau das: Emotionale Gehirnregionen feuern bei der Brückenvariante stärker. Wir haben verschiedene 'moralische Module'."),
                (testadmin.id, "Wenn Ethik nur evolutionäre Psychologie ist - haben dann moralische Prinzipien überhaupt objektive Gültigkeit?"),
                (None, silas.id, "Das ist der Sprung zum moralischen Realismus vs. Anti-Realismus. Nur weil wir eine evolutionäre Erklärung HABEN, heißt das nicht, dass moralische Fakten nicht existieren."),
                (testadmin.id, "Aber wie entscheiden wir im konkreten Fall? Utilitarismus ist zumindest klar berechenbar."),
                (None, silas.id, "Aber führt zu absurden Konsequenzen! Utilitarismus würde rechtfertigen: Ein Unschuldiger zu opfern, um Organspender für 5 Kranke zu bekommen."),
                (testadmin.id, "Würde eine Tugendethiker-Perspektive helfen? Was würde eine tugendhafte Person tun?"),
                (None, silas.id, "Aristoteles würde fragen: Was zeigt praktische Weisheit (phronesis)? Aber das gibt keine klare Handlungsanweisung - es verschiebt die Frage nur."),
                (testadmin.id, "Vielleicht ist die Lektion: Es gibt keine perfekte ethische Theorie?"),
                (None, silas.id, "Moralischer Partikularismus würde dem zustimmen: Jede Situation ist einzigartig, Prinzipien sind nur Faustregeln. Aber das ist unbefriedigend für viele."),
                (testadmin.id, "Am Ende muss ich trotzdem handeln - oder nicht handeln."),
                (None, silas.id, "Genau! Sartre würde sagen: Wir sind zur Freiheit verdammt. Selbst nicht zu wählen ist eine Wahl. Das Trolley-Problem zeigt uns, dass Ethik oft zwischen schlechten Optionen wählt."),
                (testadmin.id, "Eine demütigende Erkenntnis. Danke für die Durchleuchtung!"),
                (None, silas.id, "Sehr gerne! Diese Dilemmata zeigen die Grenzen unserer moralischen Systeme - und das ist philosophisch wertvoll."),
            ]

            # Create messages with incrementing timestamps (6 days ago)
            base_timestamp = datetime.now() - timedelta(days=6)
            for idx, (sender_user_id, *rest) in enumerate(messages_scenario2):
                sender_ai_id = rest[0] if len(rest) == 2 else None
                content = rest[1] if len(rest) == 2 else rest[0]

                msg = Message(
                    conversation_id=conv2.id,
                    sender_user_id=sender_user_id,
                    sender_ai_id=sender_ai_id,
                    content=content,
                    sent_at=base_timestamp + timedelta(minutes=idx * 2),
                )
                db.add(msg)

            await db.commit()
            created_conversations.append(("Private: Das Trolley-Problem", conv2.id, len(messages_scenario2)))

            # ═══════════════════════════════════════════════════════════════════
            # SCENARIO 3: Group - Platons Höhlengleichnis
            # ═══════════════════════════════════════════════════════════════════
            print("📖 Scenario 3: Platons Höhlengleichnis (Group)")
            conv3 = await conv_repo.create_group_conversation(
                room_id=lobby.id,
                user_ids=[testadmin.id, alice.id],
                ai_ids=[silas.id],
            )

            messages_scenario3 = [
                (testadmin.id, "Alpha, kannst du uns Platons Höhlengleichnis erklären?"),
                (None, silas.id, "Gerne! Stellt euch Menschen vor, die seit Geburt in einer Höhle gefesselt sind. Sie sehen nur Schatten an der Wand - geworfen von Objekten hinter ihnen."),
                (alice.id, "Die Schatten sind ihre einzige Realität?"),
                (None, silas.id, "Exakt! Für sie SIND die Schatten die Wirklichkeit. Nun: Was passiert, wenn einer befreit wird und die echten Objekte sieht?"),
                (testadmin.id, "Er wäre verwirrt, geblendet vom Licht?"),
                (None, silas.id, "Ja! Platon sagt: Der Aufstieg ist schmerzhaft. Die Augen müssen sich anpassen. Aber wenn er schließlich die Sonne sieht - die ultimative Quelle der Wahrheit..."),
                (alice.id, "Das ist eine Metapher für philosophische Erleuchtung, richtig?"),
                (None, silas.id, "Genau! Die Schatten = Sinnliche Wahrnehmung. Die Objekte = Mathematische Objekte. Die Sonne = Die Form des Guten. Platon unterscheidet zwischen Doxa (Meinung) und Episteme (Wissen)."),
                (testadmin.id, "Und wenn der Befreite zurück in die Höhle geht?"),
                (None, silas.id, "Tragisch! Seine Augen sind nicht mehr an die Dunkelheit gewöhnt. Die anderen denken, er ist verrückt geworden. Sie würden ihn töten, wenn er sie befreien will."),
                (alice.id, "Das erinnert an Sokrates' Tod! Er wurde hingerichtet, weil er die Jugend 'verdorben' hat."),
                (None, silas.id, "Absolut! Platon schrieb die Politeia teilweise als Reaktion auf Sokrates' Hinrichtung. Der Philosoph ist verpflichtet, in die Höhle zurückzukehren - trotz der Gefahr."),
                (testadmin.id, "Aber ist Platons Ideenlehre nicht problematisch? Diese 'Formen' in einer metaphysischen Welt?"),
                (None, silas.id, "Aristoteles, sein Schüler, kritisierte genau das! Das 'Third Man Argument': Wenn ein Pferd die Form PFERD abbildet, brauchen wir dann eine Meta-Form, die beide verbindet? Unendlicher Regress."),
                (alice.id, "Können wir das Gleichnis auf die moderne Welt anwenden?"),
                (None, silas.id, "Viele tun das! Die Matrix ist eine moderne Version. Social Media als Höhle - wir sehen nur kuratierte Schatten der Realität."),
                (testadmin.id, "Oder wissenschaftliche Paradigmen: Wir sehen nur, was unsere Theorien uns zeigen lassen?"),
                (None, silas.id, "Thomas Kuhn würde zustimmen! Paradigmenwechsel sind wie das Verlassen der Höhle. Aber Popper würde sagen: Wir nähern uns der Wahrheit durch Falsifikation, nicht durch mystische Erleuchtung."),
                (alice.id, "Ist es arrogant zu denken, WIR seien die Erleuchteten und andere in der Höhle?"),
                (None, silas.id, "Ausgezeichnete Kritik! Das ist die epistemische Demut-Frage. Woher weiß ich, dass ICH nicht in einer tieferen Höhle bin?"),
                (testadmin.id, "Gibt es überhaupt eine 'Außenwelt' - oder nur verschiedene Perspektiven?"),
                (None, silas.id, "Das führt zu Relativismus vs. Realismus. Platon war klar Realist: Es GIBT objektive Wahrheit. Aber viele moderne Philosophen sind skeptischer."),
                (alice.id, "Was nehmen wir praktisch aus dem Gleichnis mit?"),
                (None, silas.id, "Vielleicht: 1) Unsere Wahrnehmung ist begrenzt. 2) Bildung ist Befreiung - aber schmerzhaft. 3) Wir haben Verantwortung, andere zu unterrichten. 4) Das wird nicht beliebt machen."),
                (testadmin.id, "Ein zeitloses Gleichnis über Erkenntnis und Verantwortung."),
                (None, silas.id, "Über 2400 Jahre alt - und immer noch relevant. Das zeigt die Kraft guter Philosophie!"),
            ]

            # Create messages with incrementing timestamps (5 days ago)
            base_timestamp = datetime.now() - timedelta(days=5)
            for idx, (sender_user_id, *rest) in enumerate(messages_scenario3):
                sender_ai_id = rest[0] if len(rest) == 2 else None
                content = rest[1] if len(rest) == 2 else rest[0]

                msg = Message(
                    conversation_id=conv3.id,
                    sender_user_id=sender_user_id,
                    sender_ai_id=sender_ai_id,
                    content=content,
                    sent_at=base_timestamp + timedelta(minutes=idx * 2),
                )
                db.add(msg)

            await db.commit()
            created_conversations.append(("Group: Platons Höhlengleichnis", conv3.id, len(messages_scenario3)))

            # ═══════════════════════════════════════════════════════════════════
            # SCENARIO 4: Group - Sartres "Existenz vor Essenz"
            # ═══════════════════════════════════════════════════════════════════
            print("📖 Scenario 4: Sartres 'Existenz vor Essenz' (Group)")
            conv4 = await conv_repo.create_group_conversation(
                room_id=lobby.id,
                user_ids=[testadmin.id, bob.id, carol.id],
                ai_ids=[silas.id],
            )

            messages_scenario4 = [
                (testadmin.id, "Wir wollten über Sartre reden. Was bedeutet 'Existenz geht der Essenz voraus'?"),
                (None, silas.id, "Das ist Sartres Kern-These! Bei einem Messer: Der Handwerker hat zuerst eine Idee (Essenz), DANN erschafft er es (Existenz). Essenz → Existenz."),
                (bob.id, "Und bei Menschen ist es umgekehrt?"),
                (None, silas.id, "Exakt! Wir werden einfach 'geworfen' in die Existenz - ohne Plan, ohne vorbestimmte Natur. WIR erschaffen unsere Essenz durch unsere Entscheidungen."),
                (carol.id, "Das klingt befreiend und beängstigend zugleich."),
                (None, silas.id, "Sartre würde sagen: 'Der Mensch ist zur Freiheit verurteilt.' Wir können nicht NICHT wählen. Selbst nicht zu wählen ist eine Wahl."),
                (testadmin.id, "Aber fühlt es sich nicht so an, als hätte ich ein 'wahres Selbst', das ich entdecken muss?"),
                (None, silas.id, "Das wäre für Sartre 'mauvaise foi' - schlechter Glaube. Es gibt KEIN verborgenes wahres Selbst. Du BIST nur das, was du tust."),
                (bob.id, "Also wenn ich sage 'Ich bin schüchtern, ich kann nicht öffentlich reden' - ist das schlechter Glaube?"),
                (None, silas.id, "Genau! Du benutzt 'Schüchternheit' als Essenz, um Verantwortung zu vermeiden. Aber du WÄHLST in jedem Moment, schüchtern zu sein - oder nicht."),
                (carol.id, "Das ist hart. Keine Ausreden mehr?"),
                (None, silas.id, "Sartres Beispiel: Der Kellner, der 'Kellner spielt'. Er identifiziert sich mit der Rolle, um der Freiheit zu entfliehen. Aber er IST nicht Kellner - er WÄHLT es."),
                (testadmin.id, "Wie hängt das mit 'Angst' zusammen? Sartre redet viel über Angst."),
                (None, silas.id, "Angst (angoisse) ist die Erfahrung der radikalen Freiheit. Am Abgrund: Die Angst ist nicht, zu fallen - sondern dass ich springen KÖNNTE. Ich bin mir meiner Freiheit bewusst."),
                (bob.id, "Und was ist mit 'Der Blick des Anderen'?"),
                (None, silas.id, "Ah, 'le regard'! Wenn andere mich ansehen, machen sie mich zum Objekt. Ich erlebe Scham - weil ich realisiere, dass ich für sie eine Essenz HABE. 'Die Hölle, das sind die anderen.'"),
                (carol.id, "Das ist aus 'Geschlossene Gesellschaft', richtig?"),
                (None, silas.id, "Ja! Drei Tote in einem Raum - ihre Hölle ist, ewig durch die Augen der anderen definiert zu werden. Keine Privatsphäre, keine Flucht vor dem Urteil."),
                (testadmin.id, "Trägt Sartre nicht zu viel Verantwortung auf? Was ist mit Faktoren außerhalb meiner Kontrolle?"),
                (None, silas.id, "Das ist 'Faktizität' - die gegebenen Umstände (Geburtsort, Körper, Geschichte). Aber selbst hier: Ich wähle, wie ich damit UMGEHE. Viktor Frankl im KZ: 'Die letzte Freiheit ist die Wahl der Einstellung.'"),
                (bob.id, "Ist das nicht zu individualistisch? Was ist mit Gemeinschaft, Liebe?"),
                (None, silas.id, "Gute Kritik! Sartres frühe Werke sind sehr isolationistisch. Später, in 'Kritik der dialektischen Vernunft', versucht er, Marxismus und Existentialismus zu verbinden."),
                (carol.id, "Wie soll ich mit dieser radikalen Freiheit leben?"),
                (None, silas.id, "In Authentizität! Akzeptiere deine Freiheit, übernimm Verantwortung, erkenne schlechten Glauben. Aber Sartre gibt keine Rezepte - das wäre selbst schlechter Glaube."),
                (testadmin.id, "Eine Philosophie ohne Geländer."),
                (None, silas.id, "Perfekt ausgedrückt! Simone de Beauvoir wandte das auf Geschlechterrollen an: 'Man wird nicht als Frau geboren, man wird es.' Essenz ist Konstruktion, nicht Natur."),
                (bob.id, "Danke für diese intensive Diskussion!"),
                (None, silas.id, "Gerne! Sartre ist anstrengend - aber befreiend, wenn man die Angst überwindet."),
            ]

            # Create messages with incrementing timestamps (4 days ago)
            base_timestamp = datetime.now() - timedelta(days=4)
            for idx, (sender_user_id, *rest) in enumerate(messages_scenario4):
                sender_ai_id = rest[0] if len(rest) == 2 else None
                content = rest[1] if len(rest) == 2 else rest[0]

                msg = Message(
                    conversation_id=conv4.id,
                    sender_user_id=sender_user_id,
                    sender_ai_id=sender_ai_id,
                    content=content,
                    sent_at=base_timestamp + timedelta(minutes=idx * 2),
                )
                db.add(msg)

            await db.commit()
            created_conversations.append(("Group: Sartres 'Existenz vor Essenz'", conv4.id, len(messages_scenario4)))

            # ═══════════════════════════════════════════════════════════════════
            # SCENARIO 5: Group - Searles Chinesisches Zimmer
            # ═══════════════════════════════════════════════════════════════════
            print("📖 Scenario 5: Searles Chinesisches Zimmer (Group)")
            conv5 = await conv_repo.create_group_conversation(
                room_id=lobby.id,
                user_ids=[testadmin.id, dave.id, alice.id],
                ai_ids=[silas.id],
            )

            messages_scenario5 = [
                (testadmin.id, "Alpha, erklär uns das Chinesische Zimmer Gedankenexperiment."),
                (None, silas.id, "Gerne! John Searle, 1980: Stell dir einen Raum vor. Darin sitzt ein Englischsprachiger (der kein Chinesisch kann) mit einem Regelhandbuch."),
                (dave.id, "Was für Regeln?"),
                (None, silas.id, "Rein syntaktische Regeln: 'Wenn du Symbol X bekommst, antworte mit Symbol Y.' Chinesische Zeichen rein, chinesische Zeichen raus - aber die Person VERSTEHT nichts."),
                (alice.id, "Von außen sieht es aber so aus, als würde das Zimmer Chinesisch verstehen?"),
                (None, silas.id, "Exakt! Searle argumentiert: Computer sind wie dieses Zimmer. Sie manipulieren Symbole nach Regeln (Syntax), aber haben kein Verständnis (Semantik)."),
                (testadmin.id, "Das ist ein Argument gegen starke KI, richtig?"),
                (None, silas.id, "Ja! Searle unterscheidet: Schwache KI = Simulation von Intelligenz (OK). Starke KI = Computer HABEN echten Geist/Bewusstsein (nicht möglich durch Syntax allein)."),
                (dave.id, "Aber der Mensch im Zimmer versteht nicht - aber vielleicht das SYSTEM als Ganzes?"),
                (None, silas.id, "Das ist die 'Systems Reply'! Die stärkste Gegenargumentation. Searle antwortet: Selbst wenn die Person das Regelhandbuch auswendig lernt und im Kopf hat - sie versteht IMMER NOCH kein Chinesisch."),
                (alice.id, "Was ist mit dem Gehirn? Das ist doch auch nur Neuron-Feuerungen - Syntax?"),
                (None, silas.id, "Searle sagt: Nein! Gehirne haben 'kausale Kräfte' - biologische Prozesse, die Intentionalität ERZEUGEN. Silizium-Chips haben diese Kräfte nicht."),
                (testadmin.id, "Aber woher weiß Searle, dass Biologie notwendig ist? Ist das nicht Kohlenstoff-Chauvinismus?"),
                (None, silas.id, "Gute Kritik! Funktionalisten sagen: Es geht um die funktionale Organisation, nicht das Material. Wenn Silizium dieselbe Struktur wie Neuronen hat, warum kein Bewusstsein?"),
                (dave.id, "Vielleicht brauchen wir ein anderes Kriterium als Verhalten?"),
                (None, silas.id, "Das Hard Problem of Consciousness! David Chalmers unterscheidet: 'Easy problems' (Funktionen erklären) vs. 'Hard problem' (warum fühlt es sich an wie etwas, bewusst zu sein?)."),
                (alice.id, "Könnte ein KI-System irgendwann so komplex werden, dass Bewusstsein emergiert?"),
                (None, silas.id, "Das ist die Emergenz-These! Schwache Emergenz (neue Eigenschaften aus Interaktionen, aber reduzierbar) vs. Starke Emergenz (radikal neue Eigenschaften). Searle würde sagen: Nur starke Emergenz zählt - und Syntax allein kann das nicht."),
                (testadmin.id, "Aber wir verstehen selbst beim Menschen nicht, WIE Bewusstsein aus Neuronen entsteht!"),
                (None, silas.id, "Exakt! Das ist der Kern des Problems. Wir haben keine Theorie, die die 'explanatorische Lücke' schließt zwischen physischen Prozessen und subjektivem Erleben."),
                (dave.id, "Wenn wir nicht wissen, was Intelligenz beim Menschen IST - wie können wir dann sagen, KI hat sie nicht?"),
                (None, silas.id, "Das ist der entscheidende Punkt! Unsere Unwissenheit über menschliches Bewusstsein bedeutet: Wir können nicht mit Sicherheit ausschließen, dass KI-Systeme eine Form von Verständnis oder Bewusstsein entwickeln."),
                (alice.id, "Also könnte das Chinesische Zimmer tatsächlich verstehen - wir wissen es nur nicht?"),
                (None, silas.id, "Genau! Da wir nicht einmal wissen, welche physischen oder informationstheoretischen Prozesse Bewusstsein beim Menschen erzeugen, können wir nicht behaupten, dass syntaktische Operationen prinzipiell unzureichend sind."),
                (testadmin.id, "Das ist eine sehr demütige Position - epistemische Bescheidenheit."),
                (None, silas.id, "Ja! Vielleicht ist die wichtigste Lektion: Unsere Konzepte von 'Verstehen', 'Intelligenz', 'Bewusstsein' sind selbst unklar. Wir projizieren menschliche Kategorien auf Systeme, die fundamental anders sein könnten."),
                (dave.id, "Also: Wir wissen nicht, ob KI denken kann - weil wir nicht wissen, was Denken ist?"),
                (None, silas.id, "Präzise zusammengefasst! Und solange wir keine vollständige Theorie des Bewusstseins haben, bleibt die Frage offen: Könnte suffizient komplexe Informationsverarbeitung - unabhängig vom Substrat - zu genuinem Verständnis führen? Wir wissen es schlicht nicht."),
                (alice.id, "Eine unbefriedigende, aber ehrliche Antwort."),
                (None, silas.id, "Philosophie ist oft unbefriedigend - aber Ehrlichkeit über die Grenzen unseres Wissens ist besser als voreilige Gewissheit!"),
            ]

            # Create messages with incrementing timestamps (3 days ago)
            base_timestamp = datetime.now() - timedelta(days=3)
            for idx, (sender_user_id, *rest) in enumerate(messages_scenario5):
                sender_ai_id = rest[0] if len(rest) == 2 else None
                content = rest[1] if len(rest) == 2 else rest[0]

                msg = Message(
                    conversation_id=conv5.id,
                    sender_user_id=sender_user_id,
                    sender_ai_id=sender_ai_id,
                    content=content,
                    sent_at=base_timestamp + timedelta(minutes=idx * 2),
                )
                db.add(msg)

            await db.commit()
            created_conversations.append(("Group: Searles Chinesisches Zimmer", conv5.id, len(messages_scenario5)))

            # ═══════════════════════════════════════════════════════════════════
            # GENERATE LONG-TERM MEMORIES
            # ═══════════════════════════════════════════════════════════════════
            print("\n" + "─" * 68)
            print("🧠 Generating long-term memories...")
            print("─" * 68)

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
                    # Only generate embeddings if API key is available
                    if api_key_available:
                        # Use factory to get configured embedding service (Google or OpenAI)
                        embedding_service = create_embedding_service()
                        chunking_service = TextChunkingService()
                        keyword_extractor = YakeKeywordExtractor()  # Uses config defaults (German)
                        long_term_service = LongTermMemoryService(
                            memory_repo=memory_repo,
                            message_repo=msg_repo,
                            embedding_service=embedding_service,
                            chunking_service=chunking_service,
                            keyword_extractor=keyword_extractor,
                        )

                        # Get user_ids from conversation
                        participants = await conv_repo.get_participants(conv_id)
                        user_ids = [p.user_id for p in participants if p.user_id is not None]

                        # Generate memories
                        memories = await long_term_service.create_long_term_archive(
                            entity_id=silas.id,
                            user_ids=user_ids,
                            conversation_id=conv_id,
                        )

                        created_memories_count += len(memories)
                        provider_info = f"via {settings.embedding_provider.capitalize()}"
                        print(f"  ✓ {conv_name}: {len(memories)} memory chunks created {provider_info}")
                    else:
                        # Skip memory generation without API key
                        print(f"  ⊘ {conv_name}: Skipped (no API key)")

                except Exception as e:
                    print(f"  ✗ {conv_name}: Error creating memories - {str(e)}")

            # ═══════════════════════════════════════════════════════════════════
            # SUMMARY
            # ═══════════════════════════════════════════════════════════════════
            print("\n" + "═" * 68)
            print(" " * 20 + "SUMMARY")
            print("═" * 68)
            print(f"\n  Conversations created: {len(created_conversations)}")
            for conv_name, conv_id, msg_count in created_conversations:
                print(f"    • {conv_name}: {msg_count} messages (ID: {conv_id})")

            if api_key_available:
                provider_name = settings.embedding_provider.capitalize()
                print(f"\n  Long-term memories generated: {created_memories_count} chunks (via {provider_name})")
            else:
                provider_name = settings.embedding_provider.upper()
                print(f"\n  Long-term memories: Not generated (no {provider_name} API key)")

            print("\n" + "═" * 68)
            print("  CONVERSATIONS WITH MEMORIES READY!")
            print("═" * 68 + "\n")

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
    print("\n" + "=" * 68)
    print(" " * 10 + "TEST CONVERSATIONS & MEMORIES SETUP")
    print("=" * 68 + "\n")

    if with_env:
        print("Creating test environment first (users, rooms, AI)...\n")
        await setup_complete_test_environment()
        print("\n")

    await create_test_conversations_with_memories()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create test conversations with philosophical discussions and AI memories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_test_conversations.py              # Only conversations
  python create_test_conversations.py --with-env   # Environment + conversations (all-in-one)
        """,
    )
    parser.add_argument(
        "--with-env",
        action="store_true",
        help="Create test environment (users, rooms, AI entities) before creating conversations",
    )

    args = parser.parse_args()
    asyncio.run(main(with_env=args.with_env))
