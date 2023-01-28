/*
 Navicat Premium Data Transfer

 Source Server         : iotlab aliyun docker mysql
 Source Server Type    : MySQL
 Source Server Version : 80027
 Source Host           : localhost
 Source Schema         : crawler

 Target Server Type    : MySQL
 Target Server Version : 80027
 File Encoding         : 65001

 Date: 28/01/2023 17:21:28
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for newcoder_search
-- ----------------------------
DROP TABLE IF EXISTS `newcoder_search`;
CREATE TABLE `newcoder_search`  (
  `id` bigint(0) NOT NULL,
  `title` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `content` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `user` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `url` varchar(60) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `created_time` datetime(0) NOT NULL,
  `edited_time` datetime(0) NOT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
