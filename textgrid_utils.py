import logging
import numpy as np
import os
import re
import string

from typing import List

class TRFile(object):
    def __init__(self, trfilename, expectedtr=2.0045):
        """Loads data from [trfilename], should be output
        from stimulus presentation code.
        """
        self.trtimes = []
        self.soundstarttime = -1
        self.soundstoptime = -1
        self.otherlabels = []
        self.expectedtr = expectedtr
        self.frametimes = []

        if trfilename is not None:
            self.load_from_file(trfilename)

    def load_from_file(self, local_filepath):
        """Loads TR data from report with given [local_filepath].

        Parameters:

        local_filepath: str
            Path to the local TR file.
        """
        if not os.path.exists(local_filepath):
            raise FileNotFoundError(f"File not found: {local_filepath}")

        trdata_aslist = []
        for ll in open(local_filepath, encoding='utf-8'):
            trdata_aslist.append(ll)
        print(f'TRFile {local_filepath} is loaded from local path')

        # Read the report file and populate the datastructure
        for idx, ll in enumerate(trdata_aslist):
            timestr = ll.split()[0]
            label = " ".join(ll.split()[1:])
            time = float(timestr)
            if label.startswith("word") or label.startswith("START: word"):
                self.frametimes.append((time, label))
            if re.match(r'^-?\d+(?:\.\d+)$', ll.strip()):
                continue  # because some timestamps incorrectly on newline.
            if label in ("init-trigger", "trigger") or label.startswith("first"):
                self.trtimes.append(time)

            elif label == "sound-start" or label.startswith("start") or label.startswith("START"):
                self.soundstarttime = time
            elif label.split()[0] == "word" and self.soundstarttime == -1:
                self.soundstarttime = time

            elif label == "sound-stop" or label.startswith("END"):
                self.soundstoptime = time

            else:
                self.otherlabels.append((time, label))

        # Fix weird TR times
        itrtimes = np.diff(self.trtimes)
        badtrtimes = np.nonzero(itrtimes > (itrtimes.mean() * 1.5))[0]
        newtrs = []
        for btr in badtrtimes:
            print('badtrtimes are fixed')
            # Insert new TR where it was missing..
            newtrtime = self.trtimes[btr] + self.expectedtr
            newtrs.append((newtrtime, btr))

        for ntr, btr in newtrs:
            self.trtimes.insert(btr + 1, ntr)

    def simulate(self, ntrs):
        """Simulates [ntrs] TRs that occur at the expected TR.
        """
        self.trtimes = list(np.arange(ntrs) * self.expectedtr)

    def get_reltriggertimes(self):
        """Returns the times of all trigger events relative to the sound.
        """
        return np.array(self.trtimes) - self.soundstarttime

    @property
    def avgtr(self):
        """Returns the average TR for this run.
        """
        return np.diff(self.trtimes).mean()

def load_generic_trfiles(trfile_names, tr_dir="/auto/k1/huth/text/story/stimreports/generic"):
    """Loads a dictionary of generic TRFiles (i.e. not specifically from the session
    in which the data was collected.. this should be fine) for the given stories.

    Parameters:
    -----------
    trfile_names : list
        A list of trfile_names
    tr_dir : str
        Local path where trfiles are stored.

    Returns:
    --------
    trdict : dict
        A dictionary of TRFile objects for each trfile_names entry.
    """

    trdict = dict()
    for trname in trfile_names:
        try:
            trf = TRFile(os.path.join(tr_dir, trname+".report"))
            trdict[trname] = trf
        except Exception as e:
            print(e)
    return trdict

def load_textgrid_transcripts(filenames, tg_dir):
    """Loads TextGrid files and extracts word timing information.
    
    Parameters:
    -----------
    filenames : list
        A list of TextGrid filenames (without extension)
    tg_dir : str
        Local path where TextGrid files are stored.
    
    Returns:
    --------
    transcript_dict : dict
        A dictionary mapping each filename to a list of (start, end, word) tuples.
    """
    transcript_dict = {}
    
    for filename in filenames:
        filepath = os.path.join(tg_dir, filename)
        if not filepath.endswith('.TextGrid'):
            filepath += '.TextGrid'
        
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue
        
        words = []
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find the line containing "word"
        word_index = -1
        for i, line in enumerate(lines):
            if '"word"' in line or 'name = "word"' in line:
                word_index = i
                break
        
        if word_index == -1:
            print(f"Warning: 'word' tier not found in {filepath}")
            continue
        
        # Skip the next 3 lines (metadata: start, end, count)
        start_index = word_index + 4
        
        # Parse each 3-row sequence as (start, end, word)
        i = start_index
        while i + 2 < len(lines):
            try:
                start = float(lines[i].strip())
                end = float(lines[i + 1].strip())
                word = lines[i + 2].strip().strip('"')
                words.append((start, end, word))
                i += 3
            except (ValueError, IndexError):
                # Stop if we can't parse the expected format
                break
        
        transcript_dict[filename] = words
    
    return transcript_dict