import 'package:flutter/material.dart';

/// Enum representing different types of utility meters
enum UtilityType {
  gas('Gas', Icons.local_fire_department),
  water('Water', Icons.water_drop),
  electricity('Electricity', Icons.bolt);

  final String name;
  final IconData icon;

  const UtilityType(this.name, this.icon);
}

