#pragma once

// Runs diagnostic mode: captures 1s of audio, runs DSP while logging all intermediate
// arrays (STAGE 0 to 7), and then halts the system.
void pipeline_diagnostic_run();
