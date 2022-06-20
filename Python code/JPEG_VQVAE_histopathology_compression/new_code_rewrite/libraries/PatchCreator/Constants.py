# Constants file
# Used to store global constants and configuration for library scripts

from dataclasses import dataclass


@dataclass
class patch_directory_structure:
    TilesDirName: "tiles" 
    ThresholdedTilesDirName: "thresholded_tiles"  