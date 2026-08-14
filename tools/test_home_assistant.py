from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voice_assistant.config import load_assistant_config
from voice_assistant.integrations.home_assistant import (
    ALLOWED_COLOR_NAMES,
    HomeAssistantError,
    HomeAssistantClient,
    READABLE_DOMAINS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Sebastian's Home Assistant connection and discover safe entities."
    )
    parser.add_argument("--domain", choices=sorted(READABLE_DOMAINS))
    parser.add_argument("--query", help="Filter entity ID or friendly name.")
    parser.add_argument("--entity", help="Read one exact Home Assistant entity ID.")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--turn-on", action="store_true")
    actions.add_argument("--turn-off", action="store_true")
    actions.add_argument("--activate-scene", action="store_true")
    actions.add_argument("--brightness", type=int, metavar="PERCENT")
    actions.add_argument("--color", choices=sorted(ALLOWED_COLOR_NAMES))
    actions.add_argument("--color-temperature", type=int, metavar="KELVIN")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    config = load_assistant_config()
    client = HomeAssistantClient(config.home_assistant)
    try:
        connection = client.check_connection()
        print("Home Assistant connection: OK")
        if args.turn_on or args.turn_off or args.activate_scene or args.brightness is not None or args.color or args.color_temperature is not None:
            if not args.entity:
                raise ValueError("--entity is required for a control test.")
            if args.turn_on:
                result = client.turn_on_entity(args.entity)
            elif args.turn_off:
                result = client.turn_off_entity(args.entity)
            elif args.activate_scene:
                result = client.activate_scene(args.entity)
            elif args.brightness is not None:
                result = client.set_light_brightness(args.entity, args.brightness)
            elif args.color:
                result = client.set_light_color(args.entity, args.color)
            else:
                result = client.set_light_color_temperature(args.entity, args.color_temperature)
            print(json.dumps(result, indent=2))
        elif args.entity:
            print(json.dumps(client.get_entity_state(args.entity), indent=2))
        else:
            entities = client.list_entities(domain=args.domain, query=args.query)
            print(f"Matching entities: {len(entities)}")
            for entity in entities:
                capabilities = ", ".join(entity["capabilities"]) or "state only"
                print(f"{entity['entity_id']} | {entity['friendly_name']} | {entity['state']} | {capabilities}")
        return 0 if connection.get("connected") else 1
    except (HomeAssistantError, ValueError) as exc:
        print(f"Home Assistant test failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
