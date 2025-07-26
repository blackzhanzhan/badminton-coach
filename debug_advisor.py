#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试ActionAdvisor的评分问题
"""

import json
import os
from modules.action_advisor import ActionAdvisor

def debug_action_advisor():
    """调试ActionAdvisor的评分计算问题"""
    print("🔍 开始调试ActionAdvisor...")
    
    # 初始化ActionAdvisor
    advisor = ActionAdvisor()
    
    try:
        # 获取最新的staged文件
        user_file = advisor.get_latest_staged_file()
        print(f"📁 用户文件: {user_file}")
        
        # 获取模板文件
        template_file = advisor.get_template_file()
        print(f"📋 模板文件: {template_file}")
        
        # 加载数据
        print("\n📊 加载数据...")
        user_data = advisor.load_json_data(user_file)
        template_data = advisor.load_json_data(template_file)
        
        print(f"用户数据阶段数: {len(user_data)}")
        print(f"模板数据阶段数: {len(template_data)}")
        
        # 进行对比分析
        print("\n🔄 进行对比分析...")
        comparison_result = advisor.compare_stages(user_data, template_data)
        
        print(f"整体评分: {comparison_result.get('overall_score', 0):.1f}")
        print(f"关键问题数量: {len(comparison_result.get('critical_issues', []))}")
        print(f"阶段对比数量: {len(comparison_result.get('stage_comparisons', []))}")
        
        # 计算五维度评分
        print("\n📊 计算五维度评分...")
        dimension_scores = advisor.calculate_five_dimension_scores(comparison_result)
        
        print("五维度评分:")
        for dimension, score in dimension_scores.items():
            print(f"  {dimension}: {score:.1f}/100")
        
        # 生成完整报告
        print("\n📋 生成完整报告...")
        comprehensive_report = advisor.generate_comprehensive_advice()
        
        print("\n📄 完整报告内容:")
        print(f"  analysis_timestamp: {comprehensive_report.get('analysis_timestamp', 'N/A')}")
        print(f"  overall_score: {comprehensive_report.get('overall_score', 0):.1f}")
        print(f"  performance_level: {comprehensive_report.get('performance_level', 'N/A')}")
        
        # 检查dimension_scores是否在报告中
        report_dimension_scores = comprehensive_report.get('dimension_scores', {})
        print(f"\n📊 报告中的五维度评分 (数量: {len(report_dimension_scores)}):")
        if report_dimension_scores:
            for dimension, score in report_dimension_scores.items():
                print(f"  {dimension}: {score:.1f}/100")
        else:
            print("  ❌ 报告中没有dimension_scores数据！")
        
        # 检查雷达图
        radar_chart = comprehensive_report.get('radar_chart', '')
        print(f"\n📈 雷达图数据: {'存在' if radar_chart else '不存在'} (长度: {len(radar_chart) if radar_chart else 0})")
        
        # 检查其他关键字段
        print(f"\n🔍 其他关键字段:")
        print(f"  stage_analysis: {len(comprehensive_report.get('stage_analysis', []))} 个阶段")
        print(f"  critical_issues: {len(comprehensive_report.get('critical_issues', []))} 个问题")
        print(f"  detailed_suggestions: {len(comprehensive_report.get('detailed_suggestions', []))} 条建议")
        
        llm_advice = comprehensive_report.get('llm_enhanced_advice', '')
        print(f"  llm_enhanced_advice: {'存在' if llm_advice else '不存在'} (长度: {len(llm_advice) if llm_advice else 0})")
        
        # 保存调试报告
        debug_report_path = "debug_comprehensive_report.json"
        with open(debug_report_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, ensure_ascii=False, indent=2)
        print(f"\n💾 完整报告已保存到: {debug_report_path}")
        
        # 检查是否有错误字段
        if 'error' in comprehensive_report:
            print(f"\n❌ 报告中包含错误: {comprehensive_report['error']}")
        else:
            print("\n✅ 报告生成成功，无错误字段")
        
    except Exception as e:
        print(f"❌ 调试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_action_advisor()