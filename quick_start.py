#!/usr/bin/env python
"""
Mimir Quick Start Script
Gebruik dit script om snel te beginnen met de Mimir scraper.
"""

import os
import subprocess
import sys


def check_python_version():
    """Controleer of Python 3.8+ is geïnstalleerd."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 of hoger is vereist!")
        print(f"   Jouw versie: Python {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} gevonden")


def install_requirements():
    """Installeer vereiste packages."""
    print("\n📦 Installeren van vereiste packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Alle packages succesvol geïnstalleerd")
    except subprocess.CalledProcessError:
        print("❌ Fout bij installeren van packages")
        sys.exit(1)


def initialize_database():
    """Initialiseer de database."""
    print("\n🗄️ Database initialiseren...")
    try:
        subprocess.check_call([sys.executable, "db_manager.py"])
        print("✅ Database succesvol geïnitialiseerd")
    except subprocess.CalledProcessError:
        print("❌ Fout bij initialiseren database")
        sys.exit(1)


def main():
    """TODO: Add docstring."""
    """TODO: Add docstring."""
    print("🚀 MIMIR NEWS SCRAPER - QUICK START")
    print("=" * 50)

    # Controleer Python versie
    check_python_version()

    # Installeer requirements
    install_requirements()

    # Initialiseer database
    initialize_database()

    print("\n✨ Setup compleet! Wat wil je doen?\n")
    print("1. Scraper eenmalig uitvoeren")
    print("2. Scraper met scheduler starten (elke 4 uur)")
    print("3. Web interface starten")
    print("4. Statistieken bekijken")
    print("5. Email configuratie testen")
    print("6. Afsluiten")

    while True:
        choice = input("\nKies een optie (1-6): ")

        if choice == "1":
            print("\n🔄 Scraper wordt gestart...")
            subprocess.call([sys.executable, "scraper.py", "--run"])

        elif choice == "2":
            print("\n⏰ Scraper scheduler wordt gestart...")
            print("   (Druk Ctrl+C om te stoppen)")
            try:
                subprocess.call([sys.executable, "scraper.py", "--schedule"])
            except KeyboardInterrupt:
                print("\n✋ Scheduler gestopt")

        elif choice == "3":
            print("\n🌐 Web interface wordt gestart...")
            print("   Open http://localhost:5000 in je browser")
            print("   (Druk Ctrl+C om te stoppen)")
            try:
                subprocess.call([sys.executable, "web_interface.py"])
            except KeyboardInterrupt:
                print("\n✋ Web interface gestopt")

        elif choice == "4":
            print("\n📊 Statistieken:")
            subprocess.call([sys.executable, "scraper.py", "--stats"])

        elif choice == "5":
            print("\n📧 Email test wordt verzonden...")
            subprocess.call([sys.executable, "scraper.py", "--test-email"])

        elif choice == "6":
            print("\n👋 Tot ziens!")
            break

        else:
            print("❌ Ongeldige keuze, probeer opnieuw")


if __name__ == "__main__":
    main()
