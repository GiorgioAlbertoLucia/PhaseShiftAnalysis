#ifndef PHASE_SHIFTS_MAP_H
#define PHASE_SHIFTS_MAP_H

#include <string>
#include <vector>
#include <map>

struct DataPoint {
    double energy_mev;
    double value;
    double error; // 0.0 if not specified
};

/**
 * Mappings for Screenshot 2026-06-03 at 15.14.23.png
 * Contains phase shifts and mixing parameters (4.00 - 10.77 MeV)
 */
static const std::map<std::string, std::vector<DataPoint>> TABLE_15_14_23 = {
    {"1S0", {{4.00, -47.7, 6.7}, {5.51, -59.1, 0.0}, {6.82, -66.6, 0.0}, {8.82, -78.2, 8.3}, {10.77, -90.0, 0.0}}},
    {"3S1", {{4.00, -52.2, 0.0}, {5.51, -60.8, 0.0}, {6.82, -67.6, 0.0}, {8.82, -78.2, 1.4}, {10.77, -87.2, 0.0}}},
    {"3P0", {{4.00, 10.1, 7.8}, {5.51, 25.0, 0.0}, {6.82, 27.5, 0.0}, {8.82, 34.0, 5.1}, {10.77, 43.3, 0.0}}},
    {"1D2", {{4.00, -4.6, 0.0}, {5.51, -6.6, 0.0}, {6.82, -9.9, 0.0}, {8.82, -12.7, 0.0}, {10.77, -15.1, 0.0}}}
};

/**
 * Mappings for Screenshot 2026-06-03 at 15.16.40.png
 * Contains selected phase shifts (2.3 - 8.8 MeV)
 */
static const std::map<std::string, std::vector<DataPoint>> TABLE_15_16_40 = {
    {"1S0", {{2.3, -60.0, 4.5}, {8.8, -97.5, 2.5}}},
    {"3P0", {{2.3, -3.0, 5.0}, {3.0, 12.7, 6.7}, {4.5, 18.5, 7.0}, {6.8, 20.4, 5.4}, {8.8, 33.5, 4.5}}},
    {"1D2", {{8.8, -8.7, 4.3}}}
};

/**
 * Mappings for Screenshot 2026-06-03 at 15.17.51.png
 * Calculated potential parameters (1.0 - 10.0 MeV)
 * Note: Column delta_00^0 is mapped to "1S0", delta_10^1 to "3P0", etc.
 */
static const std::map<std::string, std::vector<DataPoint>> TABLE_15_17_51 = {
    {"1S0", {{1.0, -26.4, 0}, {2.0, -44.6, 0}, {3.0, -56.6, 0}, {5.0, -72.0, 0}, {10.0, -90.0, 0}}},
    {"3S1", {{1.0, -16.1, 0}, {2.0, -30.0, 0}, {3.0, -40.5, 0}, {5.0, -56.0, 0}, {10.0, -79.4, 0}}},
    {"3P0", {{1.0, 1.6, 0}, {2.0, 5.9, 0}, {3.0, 11.9, 0}, {5.0, 24.4, 0}, {10.0, 39.3, 0}}}
};

#endif