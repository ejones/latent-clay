# Latent Clay

A local-first virtual world sandbox composing AI capabilities

**This is a work in progress. Contributions welcome!**

https://github.com/ejones/latent-clay/assets/377495/ea637c4e-6fdb-4775-ac5e-cf2a0a869e2e

## Requirements

So far, this has been tested on Apple Silicon (M2 Max) but should work just as well (better even) with CUDA. At present, it requires around 20 GB VRAM at peak.

## Installation

```
git submodule update --init
python3 -m virtualenv venv
venv/bin/pip install torch==2.1.2 setuptools==69.5.1
venv/bin/pip install -r requirements.txt
```

Download the models (~5GB?)

```
venv/bin/python scripts/preload.py
```

## Running the Server

```
venv/bin/python -m latentclay.server
```

On a Mac:

```
PYTORCH_ENABLE_MPS_FALLBACK=1 venv/bin/python -m latentclay.server
```

To set a preferred GPU device (by default it will guess CUDA or Metal as available):
```
venv/bin/python -m latentclay.server --device ...
```

## Tutorial

See the video at the top and follow along with these guides.

### Create a room

1. Open the web app in your browser (http://localhost:8000 by default)
2. Pick a spot on the floor grid and click to start defining a polygonal selection
3. Click a few more points to define a simple polygon (don't cross lines)
4. Step inside the shape
5. Type <kbd>/</kbd> to open the console. Enter "room " and then whatever style of room you'd like
(e.g., "/room drawing room")
6. Hit <kbd>Enter</kbd>
7. Let the room generate; rooms (currently) take a minute or so
8. In the meantime, hit <kbd>C</kbd> to clear your selection

### Make an object

1. Once the room generates, click a point on the floor
2. Type <kbd>/</kbd> to open the console again; now it should prefill with "/create"
3. Describe your object, e.g. "lamp", and hit <kbd>Enter</kbd>
4. Type <kbd>C</kbd> to clear your selection
5. Wait for the object to generate

### Make a door

1. From the inside of a room, spot an area of wall that will be a bit wider than it is high to place
   a door
2. Click on the wall to start a wall-aligned selection and set the left edge of this region
3. Click to set the right edge of this region
4. Type <kbd>/</kbd> to open the console. Enter "door " and then whatever style of doorway you'd
   like, for instance, "/door doorway". Currently only open doorways/arches are supported (i.e., no
   hinged/sliding doors yet)
5. Type <kbd>C</kbd> to clear selection
6. Wait for the door to generate. This is the quickest of all generations.
7. If all went well, you should be able to see through the doorway to the exterior and walk through
   it.

### Create an adjoining room

1. After creating a room with a doorway, as described above, exit to the exterior
2. Turn around and define a polygonal shape selection, as above, with one edge very nearly lining up
   to the previous room
3. Create the new room as described above ("/room ...")
4. Once the room generates, type <kbd>I</kbd> to toggle door indicators. This will highlight a
   section of floor red corresponding to the doorway from the previous room (which is now covered by
   new wall)
5. Use the red door indicator as a guide to draw a corresponding doorway in the new room as
   described above
6. If all goes well, after generation, you should be able to move back and forth between the two
   rooms
   
## Keyboard Commands

- <kbd>W</kbd> <kbd>A</kbd> <kbd>S</kbd> <kbd>D</kbd> - movement
- <kbd>/</kbd> **Open Console** - to create an object, room, or door, typically after defining a
  selection with clicks
- <kbd>C</kbd> **Clear Selection**
- <kbd>I</kbd> **Toggle Door Indicators**

## Console Commands

- `/room <desc> --xz <coords>`
- `/create <desc> --x <x> --z <z> --ry [rotY] [--scale <scale>]`
- `/door <desc> --xz <coords>`
- `/undo` - removes the most recently created entity

## TODO

- improve inference performance - keep models in memory
- handle errors in UI
- removing objects
- tagging objects for reuse/defining classes
- hook up a VLM to autonomously create environments
- hook up an LLM for storytelling
- variable height rooms
- create stairs/ramps and define rooms/surfaces at current/arbitrary height
- exterior environments/terrain
- controllable weather and lighting
- load humanoids with animations
- create, define and launch projectiles
- VR client


--------

Copyright © 2024 Evan Jones
