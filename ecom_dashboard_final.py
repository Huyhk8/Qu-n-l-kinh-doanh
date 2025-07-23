import streamlit as st
import uuid
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import os

# ====== CẤU HÌNH ỨNG DỤNG ======
st.set_page_config(
    page_title="Hệ Thống Quản Lý Kinh Doanh",
    page_icon="📊",
    layout="wide"
)

# ====== TẠO DỮ LIỆU GIẢ LẬP ONLINE ======
def generate_mock_online_orders(num_orders=100):
    platforms = ['Shopee', 'Lazada', 'Tiki', 'Sendo']
    products = ['Áo thun', 'Quần jeans', 'Giày sneaker', 'Túi xách', 'Đồng hồ', 'Kính mát']
    payment_methods = ['MoMo', 'ZaloPay', 'Thẻ ngân hàng', 'COD']
    
    start_date = datetime.now() - timedelta(days=60)
    orders = []
    
    for i in range(num_orders):
        order_date = start_date + timedelta(days=np.random.randint(0, 60))
        platform = np.random.choice(platforms)
        product = np.random.choice(products)
        quantity = np.random.randint(1, 5)
        price = np.random.choice([150000, 250000, 350000, 500000])
        shipping_fee = 20000 if price > 300000 else 30000
        platform_fee = price * 0.05
        
        orders.append({
            'order_id': f'ONLINE-{1000 + i}',
            'order_date': order_date,
            'platform': platform,
            'product': product,
            'quantity': quantity,
            'price': price,
            'shipping_fee': shipping_fee,
            'platform_fee': platform_fee,
            'payment_method': np.random.choice(payment_methods),
            'status': np.random.choice(['Đã giao', 'Đang giao', 'Hủy'], p=[0.85, 0.1, 0.05]),
            'customer_id': f'CUST{np.random.randint(100, 500)}',
            'loai': 'Online'
        })
    
    return pd.DataFrame(orders)

# ====== QUẢN LÝ DỮ LIỆU OFFLINE ======
def init_offline_data():
    """Khởi tạo dữ liệu offline nếu chưa có"""
    if 'offline_transactions' not in st.session_state:
        st.session_state.offline_transactions = pd.DataFrame(columns=[
            'transaction_id', 'date', 'product', 'quantity', 
            'unit_price', 'total', 'payment_method', 'notes', 'loai'
        ])
    
    if 'offline_inventory' not in st.session_state:
        st.session_state.offline_inventory = {
            'Áo thun': 50,
            'Quần jeans': 40,
            'Giày sneaker': 30,
            'Túi xách': 25,
            'Đồng hồ': 20,
            'Kính mát': 15
        }

def save_offline_transaction(date, product, quantity, unit_price, payment_method, notes):
    """Lưu giao dịch offline mới"""
    transaction_id = f"OFFLINE-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    total = quantity * unit_price
    
    new_transaction = pd.DataFrame([{
        'transaction_id': transaction_id,
        'date': date,
        'product': product,
        'quantity': quantity,
        'unit_price': unit_price,
        'total': total,
        'payment_method': payment_method,
        'notes': notes,
        'loai': 'Offline'
    }])
    
    # Cập nhật tồn kho offline
    if product in st.session_state.offline_inventory:
        st.session_state.offline_inventory[product] -= quantity
    
    # Thêm vào lịch sử giao dịch
    st.session_state.offline_transactions = pd.concat(
        [st.session_state.offline_transactions, new_transaction], 
        ignore_index=True
    )
    
    return new_transaction

# ====== PHÂN TÍCH DỮ LIỆU TỔNG HỢP ======
def analyze_combined_data(online_df, offline_df):
    """Phân tích dữ liệu kết hợp online + offline"""
    # Kết hợp dữ liệu
    combined_df = pd.concat([online_df, offline_df], ignore_index=True)
    
    if combined_df.empty:
        return combined_df, {}
    
    # Tính toán chỉ số
    completed_df = combined_df[combined_df['status'] != 'Hủy'] if 'status' in combined_df.columns else combined_df
    
    if not completed_df.empty:
        # Tính doanh thu
        completed_df['revenue'] = completed_df.apply(
            lambda row: row['total'] if 'total' in row else (row['price'] * row['quantity']), 
            axis=1
        )
        
        # Tính lợi nhuận (giả định lợi nhuận 40% cho offline)
        completed_df['profit'] = completed_df.apply(
            lambda row: row['revenue'] * 0.4 if row['loai'] == 'Offline' 
            else (row['revenue'] - row.get('shipping_fee', 0) - row.get('platform_fee', 0)),
            axis=1
        )
    
    # Phân tích
    analysis = {
        'daily_revenue': completed_df.groupby('date')['revenue'].sum(),
        'top_products': completed_df.groupby('product')['quantity'].sum().astype(float).nlargest(5),
        'revenue_by_type': completed_df.groupby('loai')['revenue'].sum(),
        'payment_analysis': completed_df.groupby('payment_method')['revenue'].sum()
    }
    
    return completed_df, analysis

# ====== GIAO DIỆN STREAMLIT ======
def main():
    st.title("📊 Hệ Thống Quản Lý Kinh Doanh Online + Offline")
    
    # Khởi tạo dữ liệu
    init_offline_data()
    
    if 'online_orders' not in st.session_state:
        st.session_state.online_orders = generate_mock_online_orders(100)
        st.session_state.combined_df, st.session_state.analysis = analyze_combined_data(
            st.session_state.online_orders,
            st.session_state.offline_transactions
        )
    
    # Sidebar điều khiển
    with st.sidebar:
        st.header("⚙️ Điều khiển hệ thống")
        
        if st.button("🔄 Tạo dữ liệu online mới"):
            st.session_state.online_orders = generate_mock_online_orders(100)
            st.session_state.combined_df, st.session_state.analysis = analyze_combined_data(
                st.session_state.online_orders,
                st.session_state.offline_transactions
            )
            st.success("Dữ liệu online đã được cập nhật!")
        
        st.divider()
        st.subheader("Tùy chọn hiển thị")
        show_raw_data = st.checkbox("Hiển thị dữ liệu thô", False)
    
    # ====== TAB CHÍNH ======
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Tổng quan", "🛒 Kinh doanh Online", "🏪 Kinh doanh Offline", 
        "📈 Báo cáo", "📦 Kho hàng", "➕ Nhập kho"
    ])

    # Tab 5: Kho hàng
    with tab5:
        st.subheader("📦 Thống kê Kho hàng")

        if 'warehouse_data' not in st.session_state:
            st.session_state.warehouse_data = pd.DataFrame(columns=[
                'Loại sản phẩm', 'Mã sản phẩm', 'Tên sản phẩm',
                'Số lượng tồn', 'Đã vận chuyển', 'Giá nhập', 'Giá bán',
                'Tổng giá nhập', 'Tổng giá bán', 'Ngày cập nhật'
            ])

        df = st.session_state.warehouse_data.copy()
        if not df.empty:
            # Tính toán các giá trị tổng
            df['Tổng giá nhập'] = df['Số lượng tồn'] * df['Giá nhập']
            df['Tổng giá bán'] = df['Số lượng tồn'] * df['Giá bán']

            # Hiển thị dữ liệu kho hàng
            st.dataframe(df, use_container_width=True)
            
            # Biểu đồ thống kê hàng tồn theo ngày
            st.markdown("### 📊 Biểu đồ thống kê hàng tồn theo ngày")
            if 'Ngày cập nhật' in df.columns:
                df['Ngày cập nhật'] = pd.to_datetime(df['Ngày cập nhật'])
                df_date = df.groupby(df['Ngày cập nhật'].dt.date)['Số lượng tồn'].sum().reset_index()
                
                fig = px.line(
                    df_date, 
                    x='Ngày cập nhật', 
                    y='Số lượng tồn',
                    title='Biến động hàng tồn kho theo ngày',
                    markers=True
                )
                fig.update_layout(
                    xaxis_title='Ngày',
                    yaxis_title='Số lượng tồn kho',
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Tổng kết kho hàng
            st.markdown("### 📊 Tổng kết kho hàng")
            total_import_value = df['Tổng giá nhập'].sum()
            total_sale_value = df['Tổng giá bán'].sum()
            total_stock = df['Số lượng tồn'].sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Tổng giá trị nhập kho", f"{total_import_value:,.0f} VND")
            col2.metric("Tổng giá trị bán dự kiến", f"{total_sale_value:,.0f} VND")
            col3.metric("Tổng số lượng tồn", total_stock)

        st.divider()
        st.markdown("### 🛠️ Điều chỉnh thông tin tồn kho")

        with st.expander("✏️ Chỉnh sửa thông tin kho", expanded=False):
            code_input = st.text_input("🔐 Nhập mã để chỉnh sửa kho:", type="password", key="edit_code_kho_section_1_1")
            if code_input == "072025":
                product_list = st.session_state.warehouse_data['Mã sản phẩm'].unique().tolist()
                selected_code = st.selectbox("Chọn mã sản phẩm cần chỉnh sửa", product_list)
                matched_rows = st.session_state.warehouse_data[st.session_state.warehouse_data['Mã sản phẩm'] == selected_code]
                if not matched_rows.empty:
                    row = matched_rows.iloc[0]
                else:
                    st.warning("⚠️ Không tìm thấy sản phẩm với mã đã chọn.")
                    return

                cols = st.columns(3)
                current_qty = row['Số lượng tồn']
                current_import_price = row['Giá nhập']
                current_sale_price = row['Giá bán']
                
                new_qty = cols[0].number_input("Số lượng mới", value=int(current_qty), step=1)
                new_import_price = cols[1].number_input("Giá nhập mới (VND)", value=int(current_import_price), step=1000)
                new_sale_price = cols[2].number_input("Giá bán mới (VND)", value=int(current_sale_price), step=1000)

                if st.button("💾 Cập nhật thông tin", key="update_qty_btn"):
                    idx = st.session_state.warehouse_data[
                        st.session_state.warehouse_data['Mã sản phẩm'] == selected_code
                    ].index[0]
                    
                    st.session_state.warehouse_data.at[idx, 'Số lượng tồn'] = new_qty
                    st.session_state.warehouse_data.at[idx, 'Giá nhập'] = new_import_price
                    st.session_state.warehouse_data.at[idx, 'Giá bán'] = new_sale_price
                    st.session_state.warehouse_data.at[idx, 'Tổng giá nhập'] = new_qty * new_import_price
                    st.session_state.warehouse_data.at[idx, 'Tổng giá bán'] = new_qty * new_sale_price
                    st.session_state.warehouse_data.at[idx, 'Ngày cập nhật'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Lưu lịch sử chỉnh sửa
                    if "adjustment_history" not in st.session_state:
                        st.session_state.adjustment_history = []

                    st.session_state.adjustment_history.append({
                        "Thời gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Mã sản phẩm": selected_code,
                        "Tên sản phẩm": row['Tên sản phẩm'],
                        "Số lượng cũ": current_qty,
                        "Số lượng mới": new_qty,
                        "Giá nhập cũ": current_import_price,
                        "Giá nhập mới": new_import_price,
                        "Giá bán cũ": current_sale_price,
                        "Giá bán mới": new_sale_price
                    })

                    st.success("✅ Đã cập nhật thông tin kho!")

            if code_input != "" and code_input != "072025":
                st.error("❌ Mã chỉnh sửa không đúng.")

        with st.expander("📜 Xem lịch sử chỉnh sửa kho", expanded=False):
            view_code = st.text_input("🔐 Nhập mã để xem lịch sử:", type="password", key="view_code_kho_section_1_1")
            if view_code == "082025":
                if "adjustment_history" in st.session_state and st.session_state.adjustment_history:
                    st.dataframe(pd.DataFrame(st.session_state.adjustment_history))
                else:
                    st.info("📭 Chưa có chỉnh sửa nào được ghi nhận.")
            elif view_code != "":
                st.error("❌ Mã xem lịch sử không đúng.")

        if not df.empty:
            st.divider()
            st.markdown("### 📊 Tổng kết kho hàng")
            col1, col2, col3 = st.columns(3)
            col1.metric("Tổng giá trị nhập kho", f"{total_import_value:,.0f} VND")
            col2.metric("Tổng giá trị bán dự kiến", f"{total_sale_value:,.0f} VND")
            col3.metric("Tổng số lượng tồn", total_stock)
        else:
            st.info("📭 Kho hàng đang trống.")

    # Tab 6: Nhập kho
    with tab6:
        st.subheader("➕ Tạo đơn nhập kho")

        # Khởi tạo danh sách đơn hàng nếu chưa có
        if 'import_orders' not in st.session_state:
            st.session_state.import_orders = []
        if 'pending_import_orders' not in st.session_state:
            st.session_state.pending_import_orders = []

        # Nút tạo đơn hàng mới
        if st.button("➕ Tạo đơn hàng mới"):
            st.session_state.import_orders.append({
                'Loại sản phẩm': '',
                'Mã sản phẩm': '',
                'Tên sản phẩm': '',
                'Số lượng dự kiến': 0,
                'Giá nhập': 0,
                'Giá bán dự kiến': 0,
                'Nguồn hàng': '',
                'Link nguồn': '',
                'Thỏa thuận': None,
                'Hóa đơn': None,
                'Đã nhập kho': False,
                'Ngày tạo': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        # Hiển thị các đơn hàng chưa nhập kho (pending + chưa đánh dấu)
        st.markdown("### 📋 Danh sách đơn hàng chờ nhập kho")
        combined_pending = st.session_state.import_orders + st.session_state.pending_import_orders
        combined_pending = [o for o in combined_pending if not o.get('Đã nhập kho', False)]
        
        if combined_pending:
            df_pending = pd.DataFrame(combined_pending)
            total_orders = len(df_pending)
            total_qty = df_pending['Số lượng dự kiến'].sum()
            total_import_value = (df_pending['Số lượng dự kiến'] * df_pending['Giá nhập']).sum()

            st.markdown("### 📋 Danh sách đơn hàng chờ nhập kho")
            df_display = pd.DataFrame(combined_pending)
            columns_to_show = [col for col in ['Mã đơn hàng', 'Tên sản phẩm', 'Số lượng dự kiến', 'Giá nhập', 'Giá bán dự kiến', 'Ngày tạo', 'Đã nhập kho'] if col in df_display.columns]
            st.dataframe(df_display[columns_to_show])

            st.markdown("### 📊 Thống kê đơn hàng chờ nhập kho")
            col1, col2, col3 = st.columns(3)
            col1.metric("Số đơn hàng", total_orders)
            col2.metric("Tổng số lượng", total_qty)
            col3.metric("Tổng giá trị nhập dự kiến", f"{total_import_value:,.0f} VND")
        else:
            st.info("Không có đơn hàng nào đang chờ nhập kho.")

        # Hiển thị và quản lý từng đơn hàng
        for i, order in enumerate(st.session_state.import_orders):
            with st.expander(f"📦 Đơn hàng #{i+1} - {order['Tên sản phẩm']}", expanded=not order['Đã nhập kho']):
                # Thông tin cơ bản
                cols = st.columns(3)
                order['Loại sản phẩm'] = cols[0].text_input("Loại sản phẩm", value=order['Loại sản phẩm'], key=f"loai_{i}")
                order['Mã sản phẩm'] = cols[1].text_input("Mã sản phẩm", value=order['Mã sản phẩm'], key=f"ma_{i}")
                order['Tên sản phẩm'] = cols[2].text_input("Tên sản phẩm", value=order['Tên sản phẩm'], key=f"ten_{i}")

                # Số lượng và giá cả
                cols2 = st.columns(3)
                order['Số lượng dự kiến'] = cols2[0].number_input("Số lượng", 
                                                               value=order['Số lượng dự kiến'], 
                                                               min_value=0, 
                                                               step=1, 
                                                               key=f"soluong_{i}")
                order['Giá nhập'] = cols2[1].number_input("Giá nhập (VND)", 
                                                       value=order['Giá nhập'], 
                                                       min_value=0, 
                                                       step=1000, 
                                                       key=f"gianhap_{i}")
                order['Giá bán dự kiến'] = cols2[2].number_input("Giá bán dự kiến (VND)", 
                                                              value=order['Giá bán dự kiến'], 
                                                              min_value=0, 
                                                              step=1000, 
                                                              key=f"giaban_{i}")

                # Thông tin nguồn hàng
                cols3 = st.columns(2)
                order['Nguồn hàng'] = cols3[0].text_input("Nguồn hàng", value=order['Nguồn hàng'], key=f"nguon_{i}")
                order['Link nguồn'] = cols3[1].text_input("Link nguồn", value=order['Link nguồn'], key=f"link_{i}")

                # Tệp đính kèm
                order['Thỏa thuận'] = st.file_uploader("📄 Đính kèm thỏa thuận", key=f"thoa_{i}")
                order['Hóa đơn'] = st.file_uploader("🧾 Đính kèm hóa đơn", key=f"hd_{i}")

                # Trạng thái nhập kho
                order['Đã nhập kho'] = st.checkbox("✅ Đánh dấu đã nhập kho", 
                                                value=order['Đã nhập kho'], 
                                                key=f"check_{i}")

                # Nút lưu đơn hàng
                if st.button(f"💾 Lưu đơn hàng #{i+1}", key=f"save_{i}"):
                    if order['Đã nhập kho']:
                        if 'warehouse_data' not in st.session_state:
                            st.session_state.warehouse_data = pd.DataFrame(columns=[
                                'Loại sản phẩm', 'Mã sản phẩm', 'Tên sản phẩm',
                                'Số lượng tồn', 'Đã vận chuyển', 'Giá nhập', 'Giá bán',
                                'Tổng giá nhập', 'Tổng giá bán', 'Ngày cập nhật'
                            ])

                        df_kho = st.session_state.warehouse_data
                        if order['Mã sản phẩm'] in df_kho['Mã sản phẩm'].values:
                            idx = df_kho[df_kho['Mã sản phẩm'] == order['Mã sản phẩm']].index[0]
                            df_kho.at[idx, 'Số lượng tồn'] += order['Số lượng dự kiến']
                            df_kho.at[idx, 'Giá nhập'] = order['Giá nhập']
                            df_kho.at[idx, 'Giá bán'] = order['Giá bán dự kiến']
                            df_kho.at[idx, 'Tổng giá nhập'] = df_kho.at[idx, 'Số lượng tồn'] * df_kho.at[idx, 'Giá nhập']
                            df_kho.at[idx, 'Tổng giá bán'] = df_kho.at[idx, 'Số lượng tồn'] * df_kho.at[idx, 'Giá bán']
                            df_kho.at[idx, 'Ngày cập nhật'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.success(f"Cập nhật số lượng kho cho '{order['Tên sản phẩm']}'")
                        else:
                            new_row = {
                                'Loại sản phẩm': order['Loại sản phẩm'],
                                'Mã sản phẩm': order['Mã sản phẩm'],
                                'Tên sản phẩm': order['Tên sản phẩm'],
                                'Số lượng tồn': order['Số lượng dự kiến'],
                                'Đã vận chuyển': 0,
                                'Giá nhập': order['Giá nhập'],
                                'Giá bán': order['Giá bán dự kiến'],
                                'Tổng giá nhập': order['Số lượng dự kiến'] * order['Giá nhập'],
                                'Tổng giá bán': order['Số lượng dự kiến'] * order['Giá bán dự kiến'],
                                'Ngày cập nhật': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            st.session_state.warehouse_data = pd.concat(
                                [st.session_state.warehouse_data, pd.DataFrame([new_row])],
                                ignore_index=True
                            )
                            st.success(f"✅ Sản phẩm '{order['Tên sản phẩm']}' đã được thêm vào kho hàng!")

                        if order in st.session_state.import_orders:
                            st.session_state.import_orders.remove(order)
                    else:
                        st.success(f"📦 Đơn hàng #{i+1} đã được lưu tạm. Chưa nhập kho.")
    
    # Các tab khác giữ nguyên như trước...
    # Tab 1: Tổng quan
    with tab1:
        st.subheader("📈 Tổng quan kinh doanh")
        
        if st.session_state.combined_df.empty:
            st.warning("Chưa có dữ liệu kinh doanh")
            total_revenue = 0
            total_profit = 0
            online_revenue = 0
            offline_revenue = 0
        else:
            total_revenue = st.session_state.combined_df['revenue'].sum()
            total_profit = st.session_state.combined_df['profit'].sum()

            # Tính doanh thu theo loại
            revenue_by_type = st.session_state.combined_df.groupby('loai')['revenue'].sum()
            online_revenue = revenue_by_type.get('Online', 0)
            offline_revenue = revenue_by_type.get('Offline', 0)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tổng doanh thu", f"{total_revenue:,.0f} VND")
        col2.metric("Tổng lợi nhuận", f"{total_profit:,.0f} VND")
        col3.metric("Doanh thu Online", f"{online_revenue:,.0f} VND")
        col4.metric("Doanh thu Offline", f"{offline_revenue:,.0f} VND")
        
        # Biểu đồ phân bổ doanh thu
        if total_revenue > 0:
            fig_revenue_dist = px.pie(
                names=['Online', 'Offline'],
                values=[online_revenue, offline_revenue],
                title='Tỷ lệ doanh thu Online vs Offline'
            )
            st.plotly_chart(fig_revenue_dist, use_container_width=True)
    
    # Tab 2: Kinh doanh Online
    with tab2:
        st.subheader("🛒 Kinh doanh Online")
        
        if st.session_state.online_orders.empty:
            st.info("Chưa có dữ liệu kinh doanh online")
        else:
            st.dataframe(st.session_state.online_orders)
    
    # Tab 3: Kinh doanh Offline
    with tab3:
        st.subheader("🏪 Nhập liệu kinh doanh Offline")
        
        with st.form("offline_form", clear_on_submit=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                date = st.date_input("Ngày giao dịch", datetime.today())
                product = st.selectbox(
                    "Sản phẩm", 
                    options=list(st.session_state.offline_inventory.keys())
                )
                quantity = st.number_input("Số lượng", min_value=1, value=1)
                
                # Hiển thị tồn kho
                if product in st.session_state.offline_inventory:
                    st.info(f"Tồn kho hiện tại: {st.session_state.offline_inventory[product]} {product}")
            
            with col2:
                unit_price = st.number_input("Đơn giá (VND)", min_value=0, value=0)
                payment_method = st.selectbox(
                    "Phương thức thanh toán", 
                    options=["Tiền mặt", "Chuyển khoản", "Thẻ", "QR Code"]
                )
                notes = st.text_area("Ghi chú")
                
                # Tính toán tự động
                if unit_price > 0 and quantity > 0:
                    total = unit_price * quantity
                    st.metric("Tổng tiền", f"{total:,.0f} VND")
            
            submitted = st.form_submit_button("💾 Lưu giao dịch")
            
            if submitted:
                if product in st.session_state.offline_inventory:
                    if quantity > st.session_state.offline_inventory[product]:
                        st.error("⚠️ Số lượng bán vượt quá tồn kho!")
                    else:
                        new_trans = save_offline_transaction(
                            date, product, quantity, unit_price, payment_method, notes
                        )
                        st.session_state.combined_df, st.session_state.analysis = analyze_combined_data(
                            st.session_state.online_orders,
                            st.session_state.offline_transactions
                        )
                        st.success("✅ Giao dịch đã được lưu thành công!")
                else:
                    st.error("Sản phẩm không tồn tại trong kho!")
        
        st.divider()
        st.subheader("📋 Lịch sử giao dịch Offline")
        
        if not st.session_state.offline_transactions.empty:
            st.dataframe(st.session_state.offline_transactions)
        else:
            st.info("Chưa có giao dịch offline nào")
    
    # Tab 4: Báo cáo tổng hợp
    with tab4:
        st.subheader("📈 Báo cáo tổng hợp Online + Offline")
        
        if st.session_state.combined_df.empty:
            st.warning("Chưa có dữ liệu để báo cáo")
        else:
            # Báo cáo doanh thu theo ngày
            if 'daily_revenue' in st.session_state.analysis and not st.session_state.analysis['daily_revenue'].empty:
                fig_daily = px.line(
                    st.session_state.analysis['daily_revenue'].reset_index(),
                    x='date', y='revenue',
                    title='Doanh thu theo ngày',
                    labels={'date': 'Ngày', 'revenue': 'Doanh thu (VND)'}
                )
                st.plotly_chart(fig_daily, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Top sản phẩm
                if 'top_products' in st.session_state.analysis and not st.session_state.analysis['top_products'].empty:
                    fig_products = px.bar(
                        st.session_state.analysis['top_products'].reset_index(),
                        x='product', y='quantity',
                        title='Top sản phẩm bán chạy'
                    )
                    st.plotly_chart(fig_products, use_container_width=True)
            
            with col2:
                # Phân bổ doanh thu
                if 'revenue_by_type' in st.session_state.analysis and not st.session_state.analysis['revenue_by_type'].empty:
                    fig_revenue = px.pie(
                        st.session_state.analysis['revenue_by_type'].reset_index(),
                        names='loai', values='revenue',
                        title='Phân bổ doanh thu theo kênh'
                    )
                    st.plotly_chart(fig_revenue, use_container_width=True)
    
    # ====== HIỂN THỊ DỮ LIỆU THÔ ======
    if show_raw_data:
        st.subheader("📄 Dữ liệu thô tổng hợp")
        if not st.session_state.combined_df.empty:
            st.dataframe(st.session_state.combined_df)
        else:
            st.info("Chưa có dữ liệu")

# ====== CHẠY ỨNG DỤNG ======
if __name__ == "__main__":
    main()