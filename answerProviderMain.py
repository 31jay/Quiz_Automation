import json
import os
import re
from difflib import SequenceMatcher
import threading
import time

# Optional: interruptible sound alert
try:
    import pygame
    import keyboard
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False
    print("[⚠️] pygame or keyboard not installed. Sound alerts disabled.")
    print("    Install with: pip install pygame keyboard")


class AnswerProvider:
    """
    Manages answer retrieval from saved JSON database with manual fallback.
    Includes fuzzy matching and optional sound alerts.
    """

    def __init__(self, answer_file='answers.json', warning_sound='warning.mp3'):
        """
        Initialize AnswerProvider.

        Args:
            answer_file: Path to JSON file storing question-answer pairs
            warning_sound: Path to WAV/MP3 file for alert when answer not found
        """
        self.answer_file = answer_file
        self.warning_sound = warning_sound

        # Load existing answers
        if os.path.exists(answer_file):
            try:
                with open(answer_file, 'r', encoding='utf-8') as f:
                    self.answers = json.load(f)
                print(f"[✅] Loaded {len(self.answers)} answers from {answer_file}")
            except json.JSONDecodeError as e:
                print(f"[⚠️] Error loading {answer_file}: {e}")
                print("    Starting with empty answer database")
                self.answers = {}
        else:
            print(f"[ℹ️] Answer file not found. Will create {answer_file}")
            self.answers = {}

        # Initialize pygame mixer
        if SOUND_AVAILABLE:
            pygame.mixer.init()

    def _save_answers(self):
        """Save answers dictionary to JSON file."""
        try:
            with open(self.answer_file, 'w', encoding='utf-8') as f:
                json.dump(self.answers, f, indent=4, ensure_ascii=False)
            print(f"[💾] Saved answers to {self.answer_file}")
        except Exception as e:
            print(f"[❌] Error saving answers: {e}")

    def _sanitize(self, text):
        """Remove punctuation and extra whitespace for matching."""
        cleaned = re.sub(r'[^\w\s]', '', text)
        return ' '.join(cleaned.split()).lower()

    def _fuzzy_match(self, answer, options, threshold=0.7):
        """Find best matching option using fuzzy string matching."""
        best_match = None
        best_ratio = 0.0
        answer_lower = answer.lower()

        for opt in options:
            opt_lower = opt.lower()
            ratio = SequenceMatcher(None, answer_lower, opt_lower).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = opt

        if best_ratio >= threshold:
            print(f"[🔍] Fuzzy match: '{answer[:30]}...' -> "
                  f"'{best_match[:30]}...' (score: {best_ratio:.2f})")
            return best_match
        return None

    def _play_warning_sound(self, sound_file=None):
        """Play warning sound interruptibly using pygame."""
        if not SOUND_AVAILABLE:
            return

        if not sound_file:
            sound_file = self.warning_sound

        if not sound_file or not os.path.exists(sound_file):
            print(f"[⚠️] Warning sound file not found: {sound_file}")
            return

        def play_sound_interruptible():
            try:
                # Load the sound
                sound = pygame.mixer.Sound(sound_file)
                sound.play(-1)  # Loop indefinitely until stopped

                while True:
                    if keyboard.is_pressed(hotkey='esc'):
                        sound.stop()
                        print("[🔕] Warning sound stopped due to user input.")
                        break
                    time.sleep(0.05)
            except Exception as e:
                print(f"[⚠️] Could not play warning sound: {e}")

        threading.Thread(target=play_sound_interruptible, daemon=True).start()


    def _find_saved_answer(self, question):
        """Search for question in saved answers database."""
        if question in self.answers:
            return question, self.answers[question]

        sanitized_question = self._sanitize(question)
        for saved_q, saved_ans in self.answers.items():
            if self._sanitize(saved_q) == sanitized_question:
                print(f"[✅] Found answer via sanitized match")
                return saved_q, saved_ans

        return None, None

    def _prompt_manual_input(self, question, options):
        """Prompt user for manual answer selection."""
        print("\n" + "=" * 60)
        print("MANUAL INPUT REQUIRED")
        print("=" * 60)
        print(f"Question: {question[:200]}...")
        print("\n📋 Available Options:")

        for idx, opt in enumerate(options, 1):
            display_opt = opt[:100] + "..." if len(opt) > 100 else opt
            print(f"  {idx}. {display_opt}")

        print("\nEnter your choice:")
        print("  - Type option number (1-{})".format(len(options)))
        print("  - Type full or partial option text")
        print("=" * 60)

        while True:
            try:
                user_input = input("Your answer: ").strip()
                if not user_input:
                    print("[❌] Empty input. Please try again.")
                    continue

                if user_input.isdigit():
                    idx = int(user_input) - 1
                    if 0 <= idx < len(options):
                        answer = options[idx]
                        print(f"[✅] Selected option {idx + 1}: '{answer[:50]}...'")
                        return answer
                    else:
                        print(f"[❌] Invalid number. Must be 1-{len(options)}")
                        continue

                if user_input in options:
                    print(f"[✅] Exact match found: '{user_input[:50]}...'")
                    return user_input

                guess = self._fuzzy_match(user_input, options, threshold=0.5)
                if guess:
                    print(f"[🤖] Closest match found: '{guess[:50]}...'")
                    confirm = input("    Use this answer? (y/n): ").strip().lower()
                    if confirm in ['y', 'yes']:
                        return guess
                    else:
                        print("[↩️] Please try again")
                        continue

                print("[❌] No matching option found. Please try again.")
                print("    Tip: Use option number for accuracy")

            except KeyboardInterrupt:
                print("\n[❌] Input interrupted by user. Exiting program...")
                # Stop sound if any
                if SOUND_AVAILABLE:
                    pygame.mixer.stop()
                raise  # re-raise the exception to terminate the program

            except Exception as e:
                print(f"[❌] Input error: {e}")
                continue

    def get_answer(self, question, options):
        """Get answer for a question from saved database or manual input."""
        if not question:
            raise ValueError("Question cannot be empty")
        if not options:
            raise ValueError("Options list cannot be empty")

        seen = set()
        unique_options = []
        for opt in options:
            if opt not in seen:
                seen.add(opt)
                unique_options.append(opt)
        options = unique_options

        print(f"\n[🔍] Looking up answer for question...")
        print(f"     {question[:100]}...")

        saved_key, saved_answer = self._find_saved_answer(question)

        if saved_answer:
            print(f"[✅] Found saved answer: '{saved_answer[:50]}...'")
            matched_option = self._fuzzy_match(saved_answer, options, threshold=0.7)
            if matched_option:
                print(f"[✅] Matched to option: '{matched_option[:50]}...'")
                return matched_option
            else:
                print(f"[⚠️] Saved answer '{saved_answer[:50]}...' not found in current options")
                print("     Possible reasons: Options text changed / OCR differs")
                print("     Falling back to manual input...")
        else:
            print(f"[❌] No saved answer found for this question")
            self._play_warning_sound()

        answer = self._prompt_manual_input(question, options)
        print(f"[💾] Saving answer for future use...")
        self.answers[question] = answer
        self._save_answers()
        return answer

    def get_stats(self):
        """Get statistics about saved answers."""
        return {
            'total_questions': len(self.answers),
            'answer_file': self.answer_file,
            'file_exists': os.path.exists(self.answer_file)
        }

    def export_answers(self, output_file='answers_backup.json'):
        """Export answers to a backup file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.answers, f, indent=4, ensure_ascii=False)
            print(f"[✅] Exported {len(self.answers)} answers to {output_file}")
        except Exception as e:
            print(f"[❌] Export failed: {e}")

    def import_answers(self, input_file, merge=True):
        """Import answers from a file."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                imported = json.load(f)

            if merge:
                original_count = len(self.answers)
                self.answers.update(imported)
                new_count = len(self.answers) - original_count
                print(f"[✅] Imported {len(imported)} answers "
                      f"({new_count} new, {len(imported) - new_count} updated)")
            else:
                self.answers = imported
                print(f"[✅] Replaced with {len(imported)} imported answers")

            self._save_answers()
        except FileNotFoundError:
            print(f"[❌] File not found: {input_file}")
        except json.JSONDecodeError as e:
            print(f"[❌] Invalid JSON file: {e}")
        except Exception as e:
            print(f"[❌] Import failed: {e}")


# Convenience function for testing
def test_answer_provider():
    """Test the AnswerProvider functionality."""
    print("Testing AnswerProvider...")

    provider = AnswerProvider(answer_file='test_answers.json', warning_sound='warning.mp3')

    question = "What is the capital of Nepal?"
    options = ["Kathmandu", "Pokhara", "Lalitpur", "Bhaktapur"]

    print(f"\nTest Question: {question}")
    print(f"Options: {options}")

    answer = provider.get_answer(question, options)
    print(f"\nSelected Answer: {answer}")

    print("\n--- Testing retrieval ---")
    answer2 = provider.get_answer(question, options)
    print(f"Retrieved Answer: {answer2}")

    stats = provider.get_stats()
    print(f"\nStats: {stats}")


if __name__ == "__main__":
    test_answer_provider()
