# Changelog

All notable changes to this project will be documented in this file.

## [0.8.0] - 2025-09-27

### Added
- Display saved remark history in the Summary tab for each video session.

## [0.7.4] - 2025-09-27

### Fixed
- Preserved the top of zoomed frames by letting the scroll area track the full pixmap size.

## [0.7.3] - 2025-09-27

### Fixed
- Allowed the side tabs to shrink so the 80/20 splitter sizing can take effect.

## [0.7.2] - 2025-09-27

### Fixed
- Reapply the 80/20 upper splitter sizing on window resizes to keep the frame display dominant.

## [0.7.1] - 2025-09-27

### Fixed
- Enforced the 80/20 width split between the frame display and discovered videos list on initial load.

## [0.7.0] - 2025-09-27

### Changed
- Adjusted the splitter sizing so the frame display takes roughly 80% of the width and discovered videos take 20%.

## [0.6.0] - 2025-09-27

### Added
- Hide the frame/time series views when Channels, Navigation, or Summary tabs are active.
- Display discovered videos as subject ID plus filename instead of full paths.

## [0.5.0] - 2025-09-27

### Added
- Show per-subject counts of missing CSV and missing FIF files in the Summary tab table.

## [0.4.0] - 2025-09-27

### Added
- Added a remarks editor in the Summary tab and persist remarks in the per-video session YAML.

## [0.3.0] - 2025-09-27

### Added
- Subject-aware sorting for discovered videos and a dataset summary table per subject in the Summary tab.
- Dataset status aggregation by reading per-video session YAML files to count pending, ongoing, complete, and issue statuses.
