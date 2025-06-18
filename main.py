# main.py
import pandas as pd
import concurrent.futures

import data_manager
import historical_calculator
import eta_calculator

pd.options.mode.chained_assignment = None


def main():
    previous_buses_df = None
    first_time_run = True

    # --- Initial Setup ---
    print("Fetching static route and stop data...")
    stops_info_df, routes_dict = data_manager.get_routes_and_stops()

    print("Initializing historical data (if necessary)...")
    historical_calculator.initialize_historical_data(stops_info_df, routes_dict)

    print("Downloading historical data for real-time calculations...")
    distance_df = data_manager.download_distance_data()
    global_times_df = data_manager.download_global_timings()
    contextual_times_df = data_manager.download_contextualized_timings()
    routes_per_stop_map = eta_calculator.find_routes_per_stop(routes_dict)
    print("Setup complete. Starting main loop.")

    while True:
        print("\n--- Starting new ETA cycle ---")
        # Download latest bus data
        snapshot_limit = 5000 if first_time_run else 1250
        all_data_df = data_manager.download_snapshots(limit=snapshot_limit)

        # Process and calculate
        all_data_df = eta_calculator.prepare_snapshot_data(all_data_df)
        repeated_buses_df = eta_calculator.find_bus_positions(
            all_data_df, previous_buses_df, first_time_run, stops_info_df, routes_dict
        )
        repeated_buses_df, previous_buses_df = eta_calculator.process_bus_positions(
            repeated_buses_df, stops_info_df
        )

        next_bus_map = eta_calculator.find_next_bus(
            repeated_buses_df, routes_per_stop_map, routes_dict
        )

        all_stops_maps_list = eta_calculator.calculate_time_left(
            next_bus_map,
            stops_info_df,
            distance_df.copy(),
            global_times_df.copy(),
            contextual_times_df.copy(),
            routes_dict,
        )

        # Upload results
        data_manager.upload_etas(all_stops_maps_list)
        print(f"Uploaded ETAs for {len(all_stops_maps_list)} stop/route combinations.")

        first_time_run = False


if __name__ == "__main__":
    main()
